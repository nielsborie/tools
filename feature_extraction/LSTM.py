from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import Model, Sequential


def LSTM_extract(X,latent_dim=5):

    timesteps = len(X.columns)

    # Define model (based on https://blog.keras.io/building-autoencoders-in-keras.html)
    inputs = Input(shape=(timesteps, 1))
    encoded = LSTM(latent_dim)(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(1, return_sequences=True)(decoded)
    sequence_autoencoder = Model(inputs, decoded)
    sequence_encoder = Model(inputs, encoded)

    # Compile with MSE
    sequence_autoencoder.compile(
        optimizer=Adam(0.001),
        loss='mean_squared_error'
    )
    # Fitting only on sorted columns
    lstm_X = np.expand_dims(X.values, 2)

    # Fit to all data (test + train)
    sequence_autoencoder.fit(
        lstm_X, lstm_X,
        epochs=5,
        batch_size=256,
        shuffle=True,
        verbose=0,
        callbacks=[
            ReduceLROnPlateau(monitor='loss', patience=5, verbose=1),
            EarlyStopping(monitor='loss', patience=10, mode='min', min_delta=1e-5)
        ]
    )
    # Put encoded result into dataframe
    lstm_ae_df = pd.DataFrame(
        sequence_encoder.predict(lstm_X, batch_size=16), 
        columns=['lstm_AE_{}'.format(i) for i in range(5)]
    ).reset_index(drop=True)
    
    return lstm_ae_df