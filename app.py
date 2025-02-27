# app.py (versión corregida y optimizada)

"""
Estrategia de trading con KuCoin API + LightGBM para operatoria spot
Versión: 2.0 (Febrero 2025) - Integración con análisis de contexto y LLM
HORIZONTE DE PREDICCIÓN: 1 SEMANA (7 DÍAS) CON VELAS HORARIAS
"""

# --- Importaciones ---
from flask import Flask, jsonify, render_template
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import lightgbm as lgb
import pytz
import joblib
import requests
from kucoin_universal_sdk.api.client import DefaultClient
from kucoin_universal_sdk.generate.spot.market.model_get_part_order_book_req import GetPartOrderBookReqBuilder
from kucoin_universal_sdk.model.client_option import ClientOptionBuilder
from kucoin_universal_sdk.model.transport_option import TransportOptionBuilder
from kucoin_universal_sdk.model.constants import GLOBAL_API_ENDPOINT

# Importaciones para análisis de contexto
from context_analysis import get_kucoin_context_data, get_fear_and_greed_index, generate_context_report, get_crypto_news

# --- Inicialización de la aplicación Flask ---
app = Flask(__name__)

# --- Configuración de pares de trading ---
TRADING_PAIRS = [
    'BTC-USDT', 'ETH-USDT', 'THETA-USDT', 'TFUEL-USDT',
    'POL-USDT', 'LAI-USDT', 'BCH-USDT', 'WILD-USDT',
    'BNB-USDT', 'XRP-USDT', 'XLM-USDT', 'LTC-USDT',
    'SOL-USDT', 'FLUX-USDT', 'ALGO-USDT', 'TRX-USDT'
]

# --- Cargar variables de entorno ---
load_dotenv()

# --- Clase de Configuración ---
class Config:
    # Añadimos valores predeterminados para evitar errores de tipo None
    KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', '')
    KUCOIN_SECRET_KEY = os.getenv('KUCOIN_SECRET_KEY', '')
    KUCOIN_PASSPHRASE = os.getenv('KUCOIN_PASSPHRASE', '')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')  # Asegúrate de incluirlo en tu .env

# --- Cliente KuCoin ---
class KucoinClient:
    def __init__(self):
        http_transport_option = (
            TransportOptionBuilder()
            .set_keep_alive(True)
            .set_max_pool_size(10)
            .set_max_connection_per_pool(10)
            .build()
        )
        client_option = (
            ClientOptionBuilder()
            .set_key(Config.KUCOIN_API_KEY)
            .set_secret(Config.KUCOIN_SECRET_KEY)
            .set_passphrase(Config.KUCOIN_PASSPHRASE)
            .set_spot_endpoint(GLOBAL_API_ENDPOINT)
            .set_transport_option(http_transport_option)
            .build()
        )
        self.client = DefaultClient(client_option)
        self.spot_market_api = self.client.rest_service().get_spot_service().get_market_api()

    def get_ticker(self, symbol):
        """Obtiene el ticker del mercado para un símbolo específico."""
        url = f"{GLOBAL_API_ENDPOINT}/api/v1/market/orderbook/level1"
        params = {'symbol': symbol}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error al obtener ticker: {response.text}")
        data = response.json()['data']
        # Añado volumen y changeRate para compatibilidad con context_analysis.py
        return {
            'price': data.get('price', '0'),
            'size': data.get('size', '0'),
            'vol': data.get('volValue', '0'),  # Nota el cambio de 'vol' a 'volValue'
            'changeRate': data.get('changeRate', '0'),
            'time': data.get('time', 0)
        }

    def get_24hr_stats(self, symbol):
        """Obtiene estadísticas de 24 horas para un símbolo."""
        url = f"{GLOBAL_API_ENDPOINT}/api/v1/market/stats"
        params = {'symbol': symbol}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error al obtener estadísticas de 24h: {response.text}")
        return response.json()['data']

    def get_klines(self, symbol='BTC-USDT', interval='1hour', days_back=365):
        """Obtiene datos históricos de velas (klines) para un símbolo."""
        end_time = int(time.time())
        start_time = end_time - (days_back * 24 * 3600)
        url = f"{GLOBAL_API_ENDPOINT}/api/v1/market/candles"
        params = {
            'symbol': symbol,
            'type': interval,
            'startAt': start_time,
            'endAt': end_time
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error al obtener klines: {response.text}")
        data = response.json()['data']
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'close', 'high', 'low',
            'volume', 'turnover'
        ]).apply(pd.to_numeric)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.sort_values('timestamp')

    def get_order_book(self, symbol='BTC-USDT', size=20):
        """Obtiene el libro de órdenes para un símbolo."""
        try:
            request = GetPartOrderBookReqBuilder().set_symbol(symbol).set_size(str(size)).build()
            response = self.spot_market_api.get_part_order_book(request)
            
            # Verificar que bids y asks no sean None antes de procesarlos
            bids = [] if response.bids is None else [[float(b[0]), float(b[1])] for b in response.bids]
            asks = [] if response.asks is None else [[float(a[0]), float(a[1])] for a in response.asks]
            
            return {
                'bids': bids,
                'asks': asks
            }
        except Exception as e:
            print(f"Error al obtener order book para {symbol}: {e}")
            return {
                'bids': [],
                'asks': []
            }

# --- Clase para el Dataset ---
class CryptoDataset:
    def __init__(self):
        self.client = KucoinClient()

    def prepare_features(self, data):
        """Prepara las características para el modelo."""
        df = data.copy()
        df['returns'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma24'] = df['close'].rolling(window=24).mean()
        df['ma168'] = df['close'].rolling(window=168).mean()
        df['std24'] = df['close'].rolling(window=24).std()
        df['volatility'] = df['std24'] / df['close']
        df['momentum'] = df['close'] / df['close'].shift(24) - 1
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        df['target'] = df['close'].pct_change(periods=168).shift(-168) * 100
        for col in df.columns:
            if col not in ['timestamp', 'symbol', 'target']:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df.dropna()

    def _calculate_rsi(self, prices, period=14):
        """Calcula el índice de fuerza relativa (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_data(self, symbol):
        """Obtiene y prepara datos para un símbolo específico."""
        raw_data = self.client.get_klines(symbol=symbol)
        return self.prepare_features(raw_data)

    def get_all_data(self):
        """Obtiene datos para todos los pares de trading."""
        all_data = []
        for symbol in TRADING_PAIRS:
            try:
                df = self.get_data(symbol)
                df['symbol'] = symbol
                all_data.append(df)
            except Exception as e:
                print(f"Error obteniendo datos para {symbol}: {e}")
        return pd.concat(all_data, axis=0)

# --- Clase para el Modelo ---
class CryptoModel:
    def __init__(self):
        self.model = None
        self.feature_columns = [
            'returns', 'ma5', 'ma24', 'ma168', 'std24',
            'volatility', 'momentum', 'rsi', 'volume_ratio'
        ]

    def train(self, data):
        """Entrena el modelo LightGBM."""
        X = data[self.feature_columns]
        y = data['target']
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'verbose': -1
        }
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, train_data, num_boost_round=300)
        return "Modelo entrenado exitosamente"

    def predict(self, data):
        """Realiza predicciones con el modelo entrenado."""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        X = data[self.feature_columns]
        return self.model.predict(X)

# --- Generación de Señales ---
def generate_signals(predictions, current_prices):
    """Genera señales de trading basadas en las predicciones."""
    signals = []
    zona_bs_as = pytz.timezone('America/Argentina/Buenos_Aires')
    momento_actual = datetime.now(zona_bs_as)
    momento_objetivo = momento_actual + timedelta(days=7)
    texto_tiempo = momento_objetivo.strftime('%d/%m/%Y')

    # Log para depuración
    print(f"Generando señales para {len(predictions)} predicciones")
    
    for symbol, pred in predictions.items():
        # Verificar que tenemos un precio actual válido
        if symbol not in current_prices or current_prices[symbol] <= 0:
            print(f"Precio no válido para {symbol}: {current_prices.get(symbol, 'No disponible')}")
            continue
            
        try:
            abs_pred = abs(pred)
            if abs_pred > 5.0:
                categoria = "Alta probabilidad"
                confiabilidad = "Alta (>80%)"
            elif abs_pred > 2.0:
                categoria = "Probabilidad media"
                confiabilidad = "Media (50-80%)"
            elif abs_pred > 0.5:
                categoria = "Baja probabilidad"
                confiabilidad = "Baja (20-50%)"
            else:
                categoria = "Sin tendencia clara"
                confiabilidad = "Muy baja (<20%)"

            if pred > 0:
                accion = "COMPRA"
                explicacion = f"Se espera un incremento del {pred:.1f}% en el precio hasta el {texto_tiempo}"
            else:
                accion = "VENTA"
                explicacion = f"Se espera una caída del {abs(pred):.1f}% en el precio hasta el {texto_tiempo}"

            current_price = current_prices[symbol]
            expected_price = current_price * (1 + pred / 100)

            signals.append({
                "symbol": symbol,
                "señales": {
                    "qlib": {
                        "accion": accion,
                        "rentabilidad_esperada": f"{abs(pred):.1f}%",
                        "explicacion": explicacion,
                        "categoria": categoria,
                        "confiabilidad": confiabilidad,
                        "precio_actual": round(current_price, 4),
                        "precio_objetivo": round(expected_price, 4)
                    }
                }
            })
        except Exception as e:
            print(f"Error al generar señal para {symbol}: {e}")
            # Continuar con el siguiente símbolo en caso de error

    return signals

# --- Rutas de Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        dataset = CryptoDataset()
        data = dataset.get_all_data()
        model = CryptoModel()
        result = model.train(data)
        joblib.dump(model, 'crypto_model.joblib')
        return jsonify({"status": "success", "message": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict', methods=['GET'])
def api_predict():
    try:
        if not os.path.exists('crypto_model.joblib'):
            return jsonify({"status": "error", "message": "El modelo no ha sido entrenado. Por favor, entrena el modelo primero."}), 400

        model = joblib.load('crypto_model.joblib')
        dataset = CryptoDataset()
        current_data = dataset.get_all_data()

        predictions = {}
        current_prices = {}
        client = KucoinClient()

        for symbol in TRADING_PAIRS:
            symbol_data = current_data[current_data['symbol'] == symbol]
            if not symbol_data.empty:
                pred = model.predict(symbol_data.iloc[-1:])
                predictions[symbol] = float(pred[0])
            else:
                print(f"No hay datos disponibles para {symbol}")
                predictions[symbol] = 0.0

            try:
                ticker = client.get_ticker(symbol=symbol)
                current_prices[symbol] = float(ticker['price'])
            except Exception as e:
                print(f"Error al obtener precio actual para {symbol}: {e}")
                # Asignar un valor predeterminado para evitar errores
                current_prices[symbol] = 0.0

        signals = generate_signals(predictions, current_prices)

        # Filtrar señales por tipo de acción
        compra_signals = [s for s in signals if s["señales"]["qlib"]["accion"] == "COMPRA"]
        venta_signals = [s for s in signals if s["señales"]["qlib"]["accion"] == "VENTA"]

        def group_by_category(signals):
            # Debug para ver qué señales estamos recibiendo
            print(f"Agrupando {len(signals)} señales")
            
            grouped = {
                "alta_probabilidad": [],
                "probabilidad_media": [],
                "baja_probabilidad": [],
                "sin_tendencia": []
            }
            
            for signal in signals:
                try:
                    categoria = signal["señales"]["qlib"]["categoria"]
                    if "Alta" in categoria:
                        grouped["alta_probabilidad"].append(signal)
                    elif "media" in categoria:
                        grouped["probabilidad_media"].append(signal)
                    elif "Baja" in categoria:
                        grouped["baja_probabilidad"].append(signal)
                    else:
                        grouped["sin_tendencia"].append(signal)
                except KeyError as e:
                    print(f"Error al agrupar señal: {e}. Signal: {signal}")
                    
            # Debug para ver qué señales estamos enviando al frontend
            for category, items in grouped.items():
                print(f"Categoría {category}: {len(items)} señales")
                
            return grouped

        compra_grouped = group_by_category(compra_signals)
        venta_grouped = group_by_category(venta_signals)

        zona_bs_as = pytz.timezone('America/Argentina/Buenos_Aires')
        momento_actual = datetime.now(zona_bs_as)
        horizonte = momento_actual + timedelta(days=7)

        dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
        momento_str = f"{dias_semana[momento_actual.weekday()]} {momento_actual.day} de {meses[momento_actual.month - 1]} a las {momento_actual.strftime('%H:%M')}"
        horizonte_str = f"{dias_semana[horizonte.weekday()]} {horizonte.day} de {meses[horizonte.month - 1]}"

        # Protección contra errores en la obtención de datos de contexto
        try:
            # Crear una instancia específica de KucoinClient para el análisis de contexto
            # en lugar de pasar el cliente obtenido de DefaultClient
            context_client = KucoinClient()
            kucoin_context = get_kucoin_context_data(context_client, TRADING_PAIRS)
            fear_and_greed = get_fear_and_greed_index()
            
            # La función ahora devuelve tanto el informe como las noticias procesadas
            context_report, crypto_news = generate_context_report(kucoin_context, fear_and_greed, TRADING_PAIRS, predictions, Config.GEMINI_API_KEY, os.getenv('CRYPTOPANIC_API_KEY'))
            
            # Si no hay noticias procesadas, obtenerlas directamente
            if not crypto_news:
                crypto_news = get_crypto_news(os.getenv('CRYPTOPANIC_API_KEY'))
        except Exception as e:
            print(f"Error al obtener datos de contexto: {e}")
            kucoin_context = {}
            fear_and_greed = {"value": 0, "value_classification": "Desconocido"}
            context_report = "No se pudo generar el informe de contexto debido a un error."
            crypto_news = []

        return jsonify({
            "status": "success",
            "predicciones": {
                "compra": compra_grouped,
                "venta": venta_grouped,
            },
            "momento_prediccion": momento_str,
            "horizonte_prediccion": horizonte_str,
            "zona_horaria": "Buenos Aires (UTC-3)",
            "nota": "Predicciones de retorno esperado en 7 días para operatoria spot",
            "contexto": {
                "informe": context_report,
                "indice_miedo_codicia": fear_and_greed,
                "datos_kucoin": kucoin_context,
                "noticias": crypto_news
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Ejecución de la aplicación ---
#if __name__ == "__main__":
    #app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)