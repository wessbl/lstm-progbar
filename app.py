from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, LSTM
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, jsonify
import threading
import numpy as np
import yfinance as yf
import tqdm

ticker = 'AAPL'         # MSFT| DAC | AAPL
time_step = 75          # 50  | 100 |
lstm_unit = 75          # 50  | 64  | 100-200, choose one layer only
dnse_unit = 32          #  1  | 64  | choose <= 5 layers no more than 128
epochs = 20             # 10  | 50  | 100 typical max

class TrainingProgress(Callback):
    def __init__(self):
        super().__init__()
        self.progress = 0

    def on_epoch_end(self, epoch, logs=None):
        self.progress = (epoch + 1) * 100 / epochs

progress_callback = TrainingProgress()

#--- Function: Create the dataset for LSTM ---#
def create_dataset(data):
  X, y = [], []
  for i in range(len(data) - time_step - 1):
    X.append(data[i:(i + time_step), 0])
    y.append(data[i + time_step, 0])
  return np.array(X), np.array(y)
#---------------------------------------------#

def train_model(ticker):
    global progress_callback
    print("Training model...")
    model = Sequential()
    model.add(LSTM(lstm_unit, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(lstm_unit))
    model.add(Dense(dnse_unit))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    df = yf.download(ticker, '2020-01-01')
    orig_data = df['Close'].values
    orig_data = orig_data.reshape(-1, 1)    # Reshape into a 2d array: [[1], [2], [3]]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(orig_data)

    # Create Datasets to feed LSTM
    X, y = create_dataset(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train the model with the progress callback
    model.fit(X, y, epochs=epochs, callbacks=[progress_callback])

app = Flask(__name__)

# Initialize the global variable to store the callback
progress_callback = TrainingProgress()

# Show /index.html
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_training')
def start_training():
    global progress_callback
    ticker = request.args.get('ticker')
    progress_callback = TrainingProgress()  # Reset the callback
    thread = threading.Thread(target=train_model, args=(ticker,))
    thread.start()
    return jsonify({"message": "Training started"})

@app.route('/progress')
def get_progress():
    global progress_callback
    return jsonify({"progress": progress_callback.progress})

if __name__ == '__main__':
    app.run(debug=True)
