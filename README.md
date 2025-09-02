# Instala se não tiver
```bash
sudo apt update
sudo apt install python3-venv -y
```

# Cria e ativa ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

# Agora instale os pacotes
```bash
pip install -r requirements.txt
```

# Como instalar os pacotes (exemplo)
```bash
pip install "sqlalchemy>=2" "psycopg[binary]>=3" langchain_text_splitters langchain_google_genai langchain_community langchain_postgres
```

# Envie para o arquivo
```bash
pip freeze > requirements.txt
```

# Start ambiente
```bash
source venv/bin/activate
```
