class DataCleaner:
    def __init__(self):
        pass

    def clean_data(self, data_frame):
        cleaned_data = data_frame.dropna()
        return cleaned_data