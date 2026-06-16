import streamlit as st
from openai import OpenAI
import pandas as pd
import base64
from PIL import Image
import io

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
