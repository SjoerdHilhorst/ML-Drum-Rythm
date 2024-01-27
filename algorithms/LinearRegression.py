import BaseModel


class LinearRegression(BaseModel):
    def create_model(self, window, number_of_drums):
        # TODO check if this is the correct way to create a linear regression model!! AUTO-GENERATED BY GITHUB COPILOT
        """
        Creates a linear regression model.

        :param window: the size of the window of the previous barslices
        :param number_of_drums: the number of drums
        :return: a linear regression model
        """
        from keras.models import Sequential
        from keras.layers import Dense

        model = Sequential()
        model.add(Dense(number_of_drums, input_dim=window, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # TODO
