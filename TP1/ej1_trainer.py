from ej1_data_loader import Ej1DataLoader


class Ej1Trainer:
    def train(self):
        all_data = Ej1DataLoader().LoadData()
        splitted_data = self.split_data(all_data)
        train_data = splitted_data.train_data
        test_data = splitted_data.test_data
