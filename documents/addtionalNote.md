################################################

# Ollama command

################################################

##
```bash
brew install ollama
```

##
```bash
ollama serve
```

## pull model ที่ต้องการ
```bash
ollama pull llama3
```
```bash
ollama pull mistral
```
```bash
ollama pull llama3.2
```

## ส่วน embedding model

** ถ้าเปลึัยน model embedding ก๊ต้อง drop table เพราะเปลี่ยน dimension

```bash
ollama pull nomic-embed-text
```
```bash
ollama pull bge-m3 >
```
> รองรับ multilingual ได้ดีกว่า



##check
```bash
ollama list
```

##
```bash
ollama run mistral
```




```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```
