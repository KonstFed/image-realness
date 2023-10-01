import requests
import asyncio
import logging
import sys
from io import BytesIO
import json

import torch
import numpy as np
from PIL import Image

from aiogram import Bot, types, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile

with open("secret.json", "r") as f:
    secret = json.load(f)

URL = secret["url"]

TOKEN = secret["token"]
bot = Bot(token=TOKEN)
dp = Dispatcher()



@dp.message(Command("start", "help"))
async def process_start_command(message: types.Message):
    await message.reply("Чтобы воспользоваться ботом просто отправьте одно фото и он пришлёт вердикт")


@dp.message(F.photo)
async def handle_photo(msg: types.Message):
    file_id = msg.photo[-1].file_id
    file = await bot.get_file(file_id)
    result: BytesIO = await bot.download_file(file.file_path)
    # img = Image.open(result)
    # io = BytesIO()
    # img.save(io, format="jpeg")
    image_files = [("files", result)]
    response = requests.post(URL, files=image_files)
    resp = response.json()
    await bot.send_message(msg.from_user.id,  resp['msg'][0])

async def start_bot() -> None:

    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(start_bot())
