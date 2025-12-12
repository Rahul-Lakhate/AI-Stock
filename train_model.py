# train_model.py
import argparse
from utils import fetch_price_data, fetch_news_headlines, aggregate_daily_sentiment, create_features, create_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd




def build_model(input_shape):
model = Sequential()
model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
return model




def train(ticker='AAPL', period='3y', n_steps=30, save_path='models/lstm_model.h5'):
price_df = fetch_price_data(ticker, period=period)
news = fetch_news_headlines(ticker)
sentiment = aggregate_daily_sentiment(news, price_df.index)
df = create_features(price_df, sentiment)
X, y = create_sequences(df, n_steps=n_steps)
# scale features per feature dimension (scale y separately)
nsamples, nseq, nfeat = X.shape
X_reshaped = X.reshape(-1, nfeat)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(nsamples, nseq, nfeat)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1)


model = build_model(input_shape=(n_steps, nfeat))
checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='loss')
model.fit(X_scaled, y_scaled, epochs=30, batch_size=16, callbacks=[checkpoint])
# Save scalers for inference
import joblib
joblib.dump(scaler_X, 'models/scaler_X.save')
joblib.dump(scaler_y, 'models/scaler_y.save')
print('Training complete. Model saved to', save_path)




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--ticker', type=str, default='AAPL')
parser.add_argument('--period', type=str, default='3y')
parser.add_argument('--n_steps', type=int, default=30)
parser.add_argument('--save_path', type=str, default='models/lstm_model.h5')
args = parser.parse_args()
train(ticker=args.ticker, period=args.period, n_steps=args.n_steps, s
