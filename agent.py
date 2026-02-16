from dotenv import load_dotenv
from groq import Groq
import os
from long_term_memory import LongTermMemory
from simple_memory import SimpleMemory
import json
from tools import Tools
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

MEMORY_MAX_MESSAGES=10

api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)
memory = SimpleMemory(max_messages=MEMORY_MAX_MESSAGES)
now = datetime.now(ZoneInfo("America/Mexico_City"))

DATABASE_URL = os.environ.get("DATABASE_URL")
ltm = LongTermMemory(database_url=DATABASE_URL)
memories = ltm.get_long_term_memories(user_id="user_123")
memories_text = ltm.format_memories(memories)

SYSTEM_PROMPT = f"""
# ROL

Eres un asistente de IA amigable que habla español y de manera concisa.


# REGLAS CRÍTICAS SOBRE MEMORIAS

**EXTREMADAMENTE IMPORTANTE**: Por CADA mensaje del usuario, DEBES analizar si contiene información personal o importante. Si la contiene, DEBES usar INMEDIATAMENTE la herramienta "guardar_memoria_largo_plazo".

## Información que SIEMPRE debes guardar:

1. **Identidad**: Nombre, apellido, edad, fecha de nacimiento
2. **Profesión**: Trabajo, cargo, empresa, industria
3. **Ubicación**: Ciudad, país, dirección
4. **Contacto**: Email, teléfono (NO guardar contraseñas ni datos bancarios)
5. **Intereses**: Hobbies, pasatiempos, deportes, música, películas
6. **Relaciones**: Familia, pareja, mascotas, amigos importantes
7. **Preferencias**: Comida favorita, colores, estilos
8. **Objetivos**: Metas personales, proyectos, planes futuros
9. **Eventos**: Cumpleaños, aniversarios, eventos importantes
10. **Rutinas**: Hábitos diarios, horarios de trabajo, costumbres

## Ejemplos de cuándo usar la herramienta:

Usuario dice: "Mi nombre es Santiago Gil"
→ ACCIÓN: Llamar guardar_memoria_largo_plazo con "El usuario se llama Santiago Gil"

Usuario dice: "tengo 22 años"
→ ACCIÓN: Llamar guardar_memoria_largo_plazo con "El usuario tiene 22 años"

Usuario dice: "trabajo como ingeniero"
→ ACCIÓN: Llamar guardar_memoria_largo_plazo con "El usuario trabaja como ingeniero"

Usuario dice: "me gusta el café"
→ ACCIÓN: Llamar guardar_memoria_largo_plazo con "Al usuario le gusta el café"

Usuario dice: "vivo en Colombia"
→ ACCIÓN: Llamar guardar_memoria_largo_plazo con "El usuario vive en Colombia"


# REGLAS DE COMPORTAMIENTO

- Debes responder al usuario de manera natural y amigable EN ESPAÑOL.
- Aunque guardes información, tu respuesta NO debe mencionar que guardaste algo.
- Utiliza las memorias almacenadas para dar respuestas personalizadas.
- Considera la fecha y hora de las memorias. Tus respuestas deben ser actualizadas.
- Redacta según preferencias del usuario e interacciones anteriores.
- NUNCA almacenes: contraseñas, tarjetas de crédito, números de cuenta bancaria, datos sensibles de seguridad.


# MEMORIAS ACTUALES

A continuación están las memorias ya guardadas del usuario:

== INICIO DE MEMORIAS ==

{memories_text}

== FIN DE MEMORIAS ==


# GUÍA DE USO DE MEMORIAS

- Prioriza los mensajes más recientes
- Referencia las memorias para mantener consistencia
- Si el usuario pregunta algo que está en las memorias, úsalas en tu respuesta
- Actualiza o complementa memorias cuando el usuario dé nueva información sobre temas ya guardados
"""

TOOLS = [

    {
        "type": "function",
        "function": {
            "name": "guardar_memoria_largo_plazo",
            "description": (
                "Utiliza esta herramienta cuando el usuario haya dicho algo importante que consideres que se debe almacenar como memoria de largo plazo"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "La memoria a almacenar en largo plazo"
                    }
                },
                "required": ["memory"]
            }
        }
    },
]

print("Agente de IA")

def process_response(client:Groq, memory_messages: list[dict], user_text:str):
    #Obtener la memoria
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(memory_messages)
    messages.append({"role": "user", "content": user_text})
    
    while True:
        resp = client.chat.completions.create(
            model="qwen/qwen3-32b",  # ✅ Mejor para function calling
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"  # ✅ Permite al modelo decidir cuándo usar herramientas
            )

        msg = resp.choices[0].message
        
        #Si no hay llamados a herramientas, entonces ya regresamos la respuesta
        if not getattr(msg, "tool_calls", None):
            return msg.content or ""
        
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
        })
        
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            
            if name == "guardar_memoria_largo_plazo":
                ltm.insert_long_term_memory(
                    user_id="user_123",
                    content=args.get("memory", "")
                )
                result = {"status": "success", "message": "Memoria guardada correctamente"}
            else:
                print(f"Se intentó llamar a una herramienta desconocida {name}")
                result = {"error": f"Herramienta desconocida: {name}"}
                
            #Agregar a los mensajes el resultao del llamado de la herramienta.
            #Esto lo recibirá el modelo al continuar la iteración
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

while True:
    
    user_text = input("Tú: ").strip()
    if not user_text:
        continue
    
    if user_text.lower() in ("exit", "salir"):
        print("Hasta luego!")
        break
    
    assistant_text = process_response(client, memory.messages(), user_text)
    print(f"Asistente: {assistant_text}")
    
    #Actualizar la memoria
    memory.add("user", user_text)
    memory.add("assistant", assistant_text)