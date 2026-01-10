import os
from dotenv import load_dotenv
import logging
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import yfinance as yf
import pandas as pd
import asyncio
from models.models import forecast_pipeline

##################################################################################################
# Окружение
##################################################################################################
# Загрузка токена из .env
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Логгирование в консоль
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)


##################################################################################################
# Фронт
##################################################################################################
# Меню-клавиатура с понятными названиями
keyboard = [
    [KeyboardButton("Старт"), KeyboardButton("О боте")],
    [KeyboardButton("Помощь"), KeyboardButton("Прогноз и рекомендации")]
]
reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


##################################################################################################
# Бэк
##################################################################################################
# AAPL 10000
# Обработка тикера и суммы инвестиций
async def combined_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split()
    if len(parts) != 2:
        await update.message.reply_text("Недостаточно данных. Формат: ТИКЕР СУММА", reply_markup=reply_markup)
        return

    ticker = parts[0].upper()
    try:
        investment_amount = float(parts[1])
    except ValueError:
        await update.message.reply_text("Сумма указана некорректно. Используйте целые или дробные числа.", reply_markup=reply_markup)
        return

    await process_data(ticker, investment_amount, update, context)

# Загрузка исторических данных из Yahoo
async def process_data(ticker: str, investment_amount: float, update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Получаем данные за последние два года
        data_raw = yf.download(ticker, period="2y")

        # Проверка пустого ответа
        if data_raw.empty:
            await update.message.reply_text(f"К сожалению, не найдены данные по тикеру {ticker}.", reply_markup=reply_markup)
            return

        # Обработка данных
        data_close = data_raw['Close'][ticker].copy()
        data_preprocess = pd.DataFrame(data_close)
        data_preprocess['Date'] = data_preprocess.index
        data = data_preprocess.reset_index(drop=True)
        data = data.rename(columns={ticker: 'Close'})

        # Заглушка в чат
        await update.message.reply_text("Расчет метрик...", reply_markup=reply_markup)

        # Обучение моделей
        best_model_name, best_rmse, best_mape, change, buf = forecast_pipeline(data)

        # Вывод результатов
        await update.message.reply_text(f"Лучшая модель: {best_model_name}\nRMSE: {best_rmse:.2f}\nMAPE: {best_mape:.2f}", reply_markup=reply_markup)
        await update.message.reply_text(f"Разница с текущей ценой акций {ticker} через 30 дней составит: {change:.2f}%", reply_markup=reply_markup)

        # Отправляем график
        await update.message.reply_photo(photo=buf.read(), caption="Прогноз цен акций")

    except Exception as e:
        await update.message.reply_text(f"Возникла ошибка при загрузке данных: {e}", reply_markup=reply_markup)


##################################################################################################
# Функции
##################################################################################################
# /start
# Базовый старт
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Добро пожаловать! Выберите действие на клавиатуре:", reply_markup=reply_markup)

# /about
# О боте
async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text = """
Бот позволяет пользователю получать прогноз цен акций и рекомендации по торговым стратегиям. 
Необходимо ввести название компании и сумму для условной инвестиции, бот автоматически загружает исторические данные о стоимости акций, обучает несколько моделей временных рядов, выбирает наилучшую по метрикам качества и строит прогноз на ближайшие 30 дней.
"""
    await update.message.reply_text(about_text, reply_markup=reply_markup)

# /help
# Доступные команды
async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
Доступные команды:

/start - Приветствие и начало работы.
/about - Общая информация о боте.
/invest - Начать анализ стоимости ацкий и получить рекомендации.
/help - Показать это меню помощи.

Пример использования:
AAPL 10000
"""
    await update.message.reply_text(help_text, reply_markup=reply_markup)

# /invest
# Прогноз и рекомендации
async def invest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Введите тикер компании и сумму инвестиций в формате: ТИКЕР СУММА", reply_markup=reply_markup)

# Обработка нажатия кнопок или ввода тикера
async def input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    if text == "Старт":
        await start(update, context)
    elif text == "О боте":
        await about(update, context)
    elif text == "Помощь":
        await show_help(update, context)
    elif text == "Прогноз и рекомендации":
        await invest(update, context)
    else:
        await combined_input(update, context)


##################################################################################################
# Запуск
##################################################################################################
# Запуск из-под скрипта
if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()

    # /start
    application.add_handler(CommandHandler("start", start))

    # /about
    application.add_handler(CommandHandler("about", about))
    
    # /help
    application.add_handler(CommandHandler("help", show_help))

    # /invest
    application.add_handler(CommandHandler("invest", invest))

    # Обработка ввода
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, input_handler))

    # Старт бота
    application.run_polling(poll_interval=3.0)