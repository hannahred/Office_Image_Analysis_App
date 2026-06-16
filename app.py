import streamlit as st
from openai import OpenAI
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
import json

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
