import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from scipy.signal import argrelextrema
from models.models import forecast_pipeline


##################################################################################################
# Окружение
##################################################################################################
# Загрузка токена из .env
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Настройки логирования
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(filename='logs/logs.txt', filemode='a', level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)
def log_user_request(user_id, date_time, ticker, amount, best_model, metric_value, profit):
    """Функция логгирует действия пользователя в текстовом файле.
    """
    logger.warning(f"{user_id};{date_time};{ticker};{amount};{best_model};{metric_value};{profit}")


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

# Расчет инвестиционных рекомендаций
def calculate_profit(initial_investment, buy_prices, sell_prices, future_pred):
    """Функция рассчитывает итоговый баланс при следовании торговой стратегии
    """
    capital = initial_investment
    profit = 0
    shares = 0

    # Покупаем в точках минимума, продаем в точках максимума
    for buy_price, sell_price in zip(buy_prices, sell_prices):
        # Покупка на всю сумму
        num_shares = capital // buy_price
        remaining_capital = capital % buy_price
        shares += num_shares
        capital -= num_shares * buy_price

        # Продажа купленных акций
        sold_amount = num_shares * sell_price
        profit += sold_amount
        capital += sold_amount

    # Остаток капитала + стоимость оставшихся акций
    final_capital = capital + shares * future_pred[-1]
    total_profit = final_capital - initial_investment
    return total_profit

# Загрузка исторических данных из Yahoo
# Прогноз цен на акции и расчет торговой стратегии
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

        # Сравниваем модели, получаем прогнозы и метрики
        best_model_name, best_rmse, best_mape, future_pred, change, buf = forecast_pipeline(data)

        # Находим экстремумы цен
        local_max_indices = argrelextrema(future_pred, np.greater)[0]
        local_min_indices = argrelextrema(future_pred, np.less)[0]
        if (not local_max_indices.size > 0 or not local_min_indices > 0) and future_pred[0] < future_pred[-1]:
            local_min_indices = 0
            local_max_indices = future_pred[-1]
        buy_prices = future_pred[local_min_indices]
        sell_prices = future_pred[local_max_indices]

        # Рассчитываем прибыль
        total_profit = calculate_profit(investment_amount, buy_prices, sell_prices, future_pred)

        # Записываем лог
        log_user_request(
            user_id=update.effective_user.id,
            date_time=str(datetime.now()),
            ticker=ticker,
            amount=investment_amount,
            best_model=best_model_name,
            metric_value=best_rmse,
            profit=total_profit
        )

        # Формируем ответ пользователю
        summary_message = f"""📈 Прогноз цен акций для {ticker}:
⭐ Лучшая модель: {best_model_name}
🎯 Качество модели (RMSE): {best_rmse:.2f}
💨 Средняя абсолютная ошибка (MAPE): {best_mape:.2f}%

📊 Прогноз на 30 дней вперед:
🟢 Дни для покупки: {local_min_indices.tolist()}
🔴 Дни для продажи: {local_max_indices.tolist()}
↔️ Разница с текущей ценой через 30 дней: {change:.2f}%

💰 При инвестициях в размере {investment_amount:,.2f} руб. и следовании торговой стратегии, сумма итогового портфеля составит: {total_profit:,.2f} руб."""

        # Вывод ответа
        await update.message.reply_text(summary_message, reply_markup=reply_markup)
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