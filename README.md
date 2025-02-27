# 🌫️ Humocripto

Sistema de predicción de criptomonedas basado en machine learning que integra análisis técnico, contexto de mercado y noticias en tiempo real.

## ✨ Características

- 🤖 **Modelo ML Especializado**: Utiliza LightGBM para analizar patrones en velas horarias del último año
- 📊 **Predicciones a 7 días**: Análisis detallado para operatoria spot
- 🌐 **Integración con KuCoin**: Datos en tiempo real de múltiples criptomonedas
- 📰 **Análisis de Noticias**: Integración con CryptoPanic para noticias actualizadas
- 🧠 **Procesamiento LLM**: Análisis de contexto y traducción mediante modelo de lenguaje
- 📈 **Índice de Miedo y Codicia**: Visualización interactiva del sentimiento del mercado
- 🎯 **Categorización de Señales**: Clasificación por nivel de probabilidad
- 🌎 **Interfaz en Español**: Contenido completamente localizado

## 🛠️ Stack Tecnológico

- **Backend**: Python, Flask
- **ML/AI**: LightGBM, Gemini LLM
- **Frontend**: HTML5, CSS3, JavaScript
- **APIs**: KuCoin, CryptoPanic, Alternative.me
- **Análisis de Datos**: Pandas, NumPy
- **Deployment**: Render.com

## 📋 Prerequisitos

- Python 3.9+
- pip
- Acceso a API de KuCoin
- Acceso a API de CryptoPanic
- Acceso a API de Gemini

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/00nesval/humocripto
cd humocripto
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
   Crear archivo `.env` con:
```
KUCOIN_API_KEY=tu_api_key
KUCOIN_SECRET_KEY=tu_secret_key
KUCOIN_PASSPHRASE=tu_passphrase
CRYPTOPANIC_API_KEY=tu_api_key
GEMINI_API_KEY=tu_api_key
```

## 💻 Uso

1. Iniciar el servidor:
```bash
python app.py
```

2. Abrir en el navegador:
```
http://localhost:5001
```

3. Pasos de uso:
   - Entrenar el modelo (botón "Entrenar Modelo")
   - Obtener señales (botón "Obtener Señales")
   - Revisar informe de contexto
   - Consultar noticias actualizadas

## 🌐 Deploy en Render.com

1. Conectar repositorio en Render.com
2. Configurar como "Web Service"
3. Establecer variables de entorno:
   - KUCOIN_API_KEY
   - KUCOIN_SECRET_KEY
   - KUCOIN_PASSPHRASE
   - CRYPTOPANIC_API_KEY
   - GEMINI_API_KEY
4. Build command:
```
pip install -r requirements.txt
```
5. Start command:
```
gunicorn app:app
```

## 🖼️ Capturas de pantalla

[Screenshots pendientes]

## 📝 Notas importantes

- El modelo debe ser entrenado antes de obtener predicciones
- Las predicciones se basan en datos históricos y contexto actual
- El tiempo de respuesta puede variar según la carga del servidor
- Se recomienda actualizar las predicciones periódicamente

## 🔒 Seguridad

- Nunca compartir las API keys
- No almacenar keys en el código
- Usar siempre variables de entorno
- Mantener actualizado el sistema y dependencias

## 🛑 Descargo de responsabilidad

La información proporcionada tiene únicamente fines educativos e informativos. No debe interpretarse como asesoramiento financiero o recomendación de inversión. Cada persona es responsable del uso que haga de esta información y debe hacerse cargo de sus propias decisiones.

## 📜 Licencia

[MIT License](LICENSE)