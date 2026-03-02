import os
from jira import JIRA
from dotenv import load_dotenv

# 1. Загружаем данные из твоего .env
load_dotenv()

JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

def find_story_points_field():
    try:
        # Подключаемся к Jira
        jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
        print(f"✅ Connected to: {JIRA_SERVER}")
        
        # Получаем все доступные поля
        fields = jira.fields()
        
        print("\n--- Searching for Story Points Field ---")
        
        # Ищем поля, в названии которых есть 'Story Points' или 'Estimate'
        found = False
        for field in fields:
            # Jira часто называет это поле 'Story Points' или 'Story point estimate'
            if 'story point' in field['name'].lower() or 'estimate' in field['name'].lower():
                print(f"🔍 Found Field: '{field['name']}' | ID: {field['id']}")
                found = True
        
        if not found:
            print("❌ Story Points field not found. Make sure the field is enabled in your Project Settings.")
            print("\nHere are some custom fields that might be it:")
            for field in fields:
                if field['custom']:
                    print(f"Name: {field['name']} | ID: {field['id']}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    find_story_points_field()