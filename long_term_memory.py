# long_term_memory.py
import psycopg
from psycopg.rows import dict_row
import os
from dotenv import load_dotenv

class LongTermMemory:
    def __init__(self, database_url):
        self.DATABASE_URL = database_url

    # Obtener la conexión de BD de Postgres (Supabase)
    def get_conn(self):
        return psycopg.connect(self.DATABASE_URL, row_factory=dict_row)

    # Obtener las memorias de largo plazo almacenadas actualmente
    def get_long_term_memories(self, user_id: str, limit: int = 20):
        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, created_at
                FROM public.Largo_plazo
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit)
            ) 
            return cur.fetchall()

    # Insertar una nueva memoria de largo plazo
    def insert_long_term_memory(self, user_id: str, content: str):
        print(f"Inserccion de memoria llamada con {user_id}, {content}")

        content = (content or "").strip()
        if not content:
            return None

        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.Largo_plazo (user_id, content)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, content)
            )
            conn.commit()
            return cur.fetchone()

    # Obtener las memorias en un formato de texto para el System Prompt
    def format_memories(self, memories):
        if not memories:
            return "No hay memorias previas."
        
        formatted = "Memorias de largo plazo:\n"
        for idx, mem in enumerate(memories, 1):
            formatted += f"{idx}. {mem['content']} (Fecha: {mem['created_at']})\n"
        return formatted


if __name__ == "__main__":
    load_dotenv()
    
    # Verificar que DATABASE_URL existe
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ Error: DATABASE_URL no encontrada en .env")
        exit(1)
    
    print("=" * 50)
    print("DATABASE_URL cargada:")
    print(database_url)
    print("=" * 50)
    
    # Crear instancia y probar
    ltm = LongTermMemory(database_url)
    
    print("\n📖 Obteniendo memorias...")
    result = ltm.get_long_term_memories("1")
    print(f"\nSe encontraron {len(result)} memorias:\n")
    print(result)
    print(ltm.format_memories(result))

    print("\n" + "=" * 50)
    print("Memorias formateadas:")
    print("=" * 50)
    print(ltm.format_memories(result))