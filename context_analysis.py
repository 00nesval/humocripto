# context_analysis.py

import requests
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")

def get_kucoin_context_data(client, symbols):
    """
    Obtiene datos de contexto de KuCoin para los símbolos especificados.
    
    Args:
        client: Una instancia de KucoinClient (NO DefaultClient)
        symbols: Lista de símbolos para obtener datos
        
    Returns:
        Diccionario con datos de contexto para cada símbolo
    """
    context_data = {}
    for symbol in symbols:
        try:
            # Obtener libro de órdenes
            order_book = client.get_order_book(symbol=symbol, size=20)
            bid_depth = sum([b[1] for b in order_book['bids']])
            ask_depth = sum([a[1] for a in order_book['asks']])
            depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 0

            # Obtener estadísticas de 24 horas
            stats = client.get_ticker(symbol)
            
            # Depuración para ver qué devuelve realmente la API
            print(f"Datos de ticker para {symbol}: {stats}")
            
            # Intenta obtener datos de 24 horas directamente de la API específica
            try:
                stats_24h = client.get_24hr_stats(symbol)
                print(f"Estadísticas 24h para {symbol}: {stats_24h}")
                volume_24h = float(stats_24h.get('volValue', stats_24h.get('vol', 0)))
                price_change_24h = float(stats_24h.get('changeRate', 0)) * 100
            except Exception as e:
                print(f"Error al obtener estadísticas 24h, usando datos del ticker: {e}")
                # Verificar que el valor existe o usar un valor predeterminado
                volume_24h = float(stats.get('vol', stats.get('volValue', 0)))
                price_change_24h = float(stats.get('changeRate', 0)) * 100

            context_data[symbol] = {
                'order_book': {
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth,
                    'depth_ratio': depth_ratio,
                },
                '24h_stats': {
                    'volume': volume_24h,
                    'price_change_pct': price_change_24h,
                }
            }
        except Exception as e:
            print(f"Error al obtener datos de contexto para {symbol}: {e}")
            context_data[symbol] = {
                'order_book': {
                    'bid_depth': 0,
                    'ask_depth': 0,
                    'depth_ratio': 0,
                },
                '24h_stats': {
                    'volume': 0,
                    'price_change_pct': 0,
                }
            }
    return context_data
def get_fear_and_greed_index():
    """
    Obtiene el Índice de Miedo y Codicia de Alternative.me.
    
    Returns:
        Diccionario con el valor y clasificación del índice
    """
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]
        
        # Traducir value_classification al español
        classification_en = data['value_classification']
        classification_es = traducir_clasificacion(classification_en)
        
        return {
            'value': int(data['value']),
            'value_classification': classification_es,
            'timestamp': data['timestamp'],
            'time_until_update': data['time_until_update']
        }
    except Exception as e:
        print(f"Error al obtener el Índice de Miedo y Codicia: {e}")
        return {
            'value': 50,  # Valor neutral por defecto
            'value_classification': "Desconocido",
            'timestamp': str(int(datetime.now().timestamp())),
            'time_until_update': "Desconocido"
        }
        
def traducir_clasificacion(texto_en):
    """
    Traduce la clasificación del índice de miedo y codicia del inglés al español.
    
    Args:
        texto_en: Texto en inglés
        
    Returns:
        Texto traducido al español
    """
    traducciones = {
        "Extreme Fear": "Miedo Extremo",
        "Fear": "Miedo",
        "Neutral": "Neutral",
        "Greed": "Codicia",
        "Extreme Greed": "Codicia Extrema"
    }
    
    return traducciones.get(texto_en, texto_en)  # Si no hay traducción, devolver el original

def traducir_texto(texto):
    """
    Traduce un texto del inglés al español.
    Implementación simple para términos comunes de criptomonedas.
    
    Args:
        texto: Texto en inglés a traducir
        
    Returns:
        Texto traducido al español
    """
    if not texto:
        return ""
    
    # Diccionario de términos comunes en criptomonedas
    terminos = {
        "Bitcoin": "Bitcoin",
        "Ethereum": "Ethereum",
        "crypto": "cripto",
        "cryptocurrency": "criptomoneda",
        "cryptocurrencies": "criptomonedas",
        "blockchain": "blockchain",
        "token": "token",
        "tokens": "tokens",
        "market": "mercado",
        "markets": "mercados",
        "bull": "alcista",
        "bear": "bajista",
        "bullish": "alcista",
        "bearish": "bajista",
        "rally": "repunte",
        "crash": "caída",
        "dump": "caída",
        "pump": "subida",
        "mining": "minería",
        "price": "precio",
        "value": "valor",
        "exchange": "exchange",
        "trading": "trading",
        "trader": "trader",
        "traders": "traders",
        "altcoin": "altcoin",
        "altcoins": "altcoins",
        "stablecoin": "stablecoin",
        "stablecoins": "stablecoins",
        "DeFi": "DeFi",
        "NFT": "NFT",
        "NFTs": "NFTs",
        "smart contract": "contrato inteligente",
        "smart contracts": "contratos inteligentes",
        "ICO": "ICO",
        "airdrop": "airdrop",
        "whale": "ballena",
        "whales": "ballenas",
        "HODL": "HODL",
        "ATH": "ATH",
        "ATL": "ATL",
        "FUD": "FUD",
        "FOMO": "FOMO",
        "ROI": "ROI"
    }
    
    # Reemplazar términos conocidos
    for en, es in terminos.items():
        texto = texto.replace(en, es)
    
    return texto

def get_crypto_news(api_key=CRYPTOPANIC_API_KEY, limit=5, days=2):
    """
    Obtiene las noticias más recientes relacionadas con criptomonedas desde CryptoPanic.
    Traduce los títulos y descripciones al español usando Google Translate.
    Filtra las noticias para mostrar solo las de los últimos días especificados.
    
    Args:
        api_key: API Key para CryptoPanic
        limit: Número de noticias a obtener (por defecto 5)
        days: Número de días hacia atrás para filtrar noticias (por defecto 2)
        
    Returns:
        Lista de diccionarios con información de las noticias traducidas
    """
    try:
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": api_key,
            "public": "true",
            "filter": "hot",
            "kind": "news",
            "limit": limit
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Calcular la fecha límite para filtrar noticias (hace 'days' días)
        from datetime import datetime, timedelta
        date_limit = datetime.now() - timedelta(days=days)
        
        news_list = []
        for post in data.get("results", []):
            # Verificar la fecha de publicación
            published_at = post.get("published_at", "")
            if published_at:
                try:
                    # Convertir la fecha de publicación a objeto datetime
                    published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    # Filtrar noticias más antiguas que el límite de días
                    if published_date < date_limit:
                        continue
                except Exception as e:
                    print(f"Error al procesar fecha de publicación: {e}")
            
            # Obtener datos originales - ya no traducimos aquí, dejamos que el LLM lo haga
            title_en = post.get("title", "")
            
            # Aplicar traducción básica para el título (solo para mostrar algo mientras el LLM procesa)
            title_es = traducir_texto(title_en)
            
            # Obtener la URL original de la fuente
            # Según la documentación de CryptoPanic, la URL original está en source.url
            original_url = post.get("source", {}).get("url", "")
            if not original_url:
                # Si no hay URL original en source.url, intentar usar la URL directa del post
                original_url = post.get("url", "")
            
            news_item = {
                "title": title_es,  # Título con traducción básica para mostrar algo inicialmente
                "title_original": title_en,  # Título original para que el LLM lo traduzca
                "url": original_url,
                "published_at": published_at,
                "source": post.get("source", {}).get("title", ""),
                "source_domain": post.get("source", {}).get("domain", ""),
                "votes": post.get("votes", {}),
                "currencies": [c.get("code", "") for c in post.get("currencies", [])]
            }
            news_list.append(news_item)
            
        return news_list
    except Exception as e:
        print(f"Error al obtener noticias de CryptoPanic: {e}")
        return []

def generate_context_report(kucoin_data, fear_and_greed, trading_pairs, model_predictions, api_key, news_api_key=CRYPTOPANIC_API_KEY):
    """
    Genera un informe de contexto con el LLM.
    
    Args:
        kucoin_data: Datos de contexto de KuCoin
        fear_and_greed: Índice de Miedo y Codicia
        trading_pairs: Lista de pares de trading
        model_predictions: Predicciones del modelo
        api_key: API Key de Gemini
        
    Returns:
        Informe de contexto generado
    """
    # Preparar datos de KuCoin para el prompt
    kucoin_prompt = ""
    for symbol in trading_pairs:
        if symbol in kucoin_data and kucoin_data[symbol]:
            data = kucoin_data[symbol]
            kucoin_prompt += f"**{symbol}:**\n"
            kucoin_prompt += f"- Profundidad (compra/venta): {data['order_book']['depth_ratio']:.2f}\n"
            kucoin_prompt += f"- Volumen (24h): {data['24h_stats']['volume']:.2f}\n"
            kucoin_prompt += f"- Cambio de precio (24h): {data['24h_stats']['price_change_pct']:.2f}%\n"
        else:
            kucoin_prompt += f"**{symbol}:**\n- No hay datos disponibles\n"

    # Preparar datos de predicciones para el prompt
    predictions_prompt = ""
    for symbol, pred in model_predictions.items():
        predictions_prompt += f"- {symbol}: {pred:.2f}%\n"

    # Obtener noticias de CryptoPanic
    crypto_news = get_crypto_news(news_api_key)
    news_prompt = ""
    
    # Preparar prompt para resumir noticias
    news_summary_prompt = ""
    if crypto_news:
        news_prompt = "Noticias recientes del mercado cripto:\n"
        news_summary_prompt = "Además, genera un resumen de 3 líneas para cada una de las siguientes noticias de criptomonedas:\n\n"
        
        for i, news in enumerate(crypto_news, 1):
            news_prompt += f"{i}. {news['title']}\n"
            news_prompt += f"   Fuente: {news['source']} ({news['source_domain']})\n\n"
            
            # Agregar la noticia al prompt de resumen
            news_summary_prompt += f"Noticia {i}: {news['title_original']}\n"
            news_summary_prompt += "\n"
    
    # Construir el prompt completo
    prompt = f"""
    Genera un informe estructurado en 4 secciones sobre el mercado de criptomonedas actual. Formatea cada sección con su título en una línea y su contenido en párrafos separados.

    $[Estado actual]$
    Basate en el Índice de Miedo y Codicia que actualmente muestra un valor de {fear_and_greed['value']} ({fear_and_greed['value_classification']}) para explicar el estado emocional actual del mercado.

    $[Señales del mercado]$
    Describe brevemente si predominan compradores o vendedores, movimientos significativos y volumen.

    $[Panorama cripto]$
    Toma como base las noticias recientes proporcionadas para brindar una síntesis detallada de las principales situaciones y movimientos que están influyendo en el rumbo del mercado cripto. Este eje debe ser el más extenso y analítico.

    $[Perspectiva]$
    Tomando como base lo planteado en las secciones anteriores, indica qué esperar en los próximos días.

    IMPORTANTE:
    - Usa $[Título]$ para los encabezados de cada sección
    - Evita duplicar información entre el título y el contenido
    - Usa lenguaje claro y directo, evitando tecnicismos
    - El eje "Panorama cripto" debe ser el más extenso y detallado

    Datos KuCoin:
    {kucoin_prompt}

    Predicciones (solo referencia):
    {predictions_prompt}

    {news_prompt}

    {news_summary_prompt}
    Para cada noticia, traduce el título al español y genera un resumen en español de 4 líneas que capture la esencia de la noticia. El resumen debe ser conciso pero informativo, proporcionando los detalles más relevantes.
    
    IMPORTANTE:
    - Traduce TODOS los títulos al español.
    - No menciones números de noticias en el informe principal.
    - Si necesitas hacer referencia a una noticia en el informe, usa su tema principal en lugar de su número.
    - Formatea cada resumen como: "RESUMEN_NOTICIA_X: [Título traducido en español] - [Resumen de 4 líneas]" donde X es el número de la noticia.
    """

    # Realizar petición a la API de Gemini con el nuevo modelo
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro-exp-02-05:generateContent"
        headers = {'Content-Type': 'application/json'}
        
        # Configuración actualizada según el ejemplo proporcionado
        data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generation_config": {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain"
            }
        }

        # Aumentamos el timeout a 60 segundos para evitar errores de tiempo agotado
        response = requests.post(f"{url}?key={api_key}", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        # Procesar la respuesta de la API
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            response_text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Extraer los resúmenes de noticias
            news_summaries = []
            for i, news in enumerate(crypto_news, 1):
                summary_marker = f"RESUMEN_NOTICIA_{i}:"
                if summary_marker in response_text:
                    # Extraer el resumen
                    start_idx = response_text.find(summary_marker) + len(summary_marker)
                    end_idx = response_text.find(f"RESUMEN_NOTICIA_{i+1}:", start_idx) if i < len(crypto_news) else len(response_text)
                    if end_idx == -1:  # Si no se encuentra el siguiente marcador
                        end_idx = len(response_text)
                    
                    full_summary = response_text[start_idx:end_idx].strip()
                    
                    # Extraer el título traducido y el resumen
                    if ' - ' in full_summary:
                        title_part, summary_part = full_summary.split(' - ', 1)
                        # Actualizar el título con la traducción y guardar el resumen
                        news['title'] = title_part.strip()
                        news['summary'] = summary_part.strip()
                    else:
                        news['summary'] = full_summary
                else:
                    news['summary'] = "No se pudo generar un resumen para esta noticia."
            
            # Eliminar los resúmenes del informe principal
            # Buscar diferentes variantes de marcadores de resúmenes
            resumen_markers = ["RESÚMENES DE NOTICIAS:", "RESÚMENES DE NOTICIAS", "RESUMEN DE NOTICIAS:", "RESUMEN DE NOTICIAS"]
            
            for marker in resumen_markers:
                if marker in response_text:
                    response_text = response_text.split(marker)[0].strip()
                    break
            
            # Si no se encontró ningún marcador específico, buscar por cada resumen individual
            if any(f"RESUMEN_NOTICIA_{i}:" in response_text for i in range(1, len(crypto_news) + 1)):
                for i in range(len(crypto_news), 0, -1):
                    summary_marker = f"RESUMEN_NOTICIA_{i}:"
                    if summary_marker in response_text:
                        start_idx = response_text.find(summary_marker)
                        response_text = response_text[:start_idx].strip()
                        break
            
            # Eliminar líneas vacías al final
            response_text = response_text.rstrip()
            
            # Devolver el informe y las noticias con resúmenes
            return response_text, crypto_news
        else:
            print(f"Respuesta inesperada de Gemini: {json.dumps(result)}")
            return "No se pudo generar el informe de contexto. Respuesta inesperada del modelo.", crypto_news
            
    except requests.exceptions.RequestException as e:
        print(f"Error de red al generar el informe con el LLM: {e}")
        return "No se pudo generar el informe de contexto debido a un error de red.", crypto_news
    except ValueError as e:  # Para errores de JSON
        print(f"Error de formato al generar el informe con el LLM: {e}")
        return "No se pudo generar el informe de contexto debido a un error de formato.", crypto_news
    except Exception as e:
        print(f"Error desconocido al generar el informe con el LLM: {e}")
        return "No se pudo generar el informe de contexto debido a un error desconocido.", crypto_news