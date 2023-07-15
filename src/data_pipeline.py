from os import path
import torchdata.datapipes as dp
import tiktoken
from torchdata.dataloader2 import DataLoader2

class Preprocessor:
    def __init__(
            self,
            dataset,
            batch_size=32,
            cntx_len=32,
            model_encoding='gpt2',
            shuffle=True,
            val_data_path=None,
            data_field='content',
            test_size=.005):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cntx_len = cntx_len
        self.model_encoding = model_encoding
        self.shuffle = shuffle
        self.val_data_path = val_data_path
        self.data_field = data_field
        self.test_size = test_size

    def _encode_data(self, data):
        enc = tiktoken.encoding_for_model(self.model_encoding)
        return enc.encode(data, allowed_special='all')

    def _get_content(self, data): 
        return data[self.data_field]

    def _causal_pipe(self, reader_dp):
        tokens = reader_dp.flatmap(self._encode_data)
        batch = tokens.batch(batch_size=self.cntx_len+1, drop_last = True)
        x_dp, y_dp = batch.shuffle().fork(num_instances = 2) if self.shuffle else batch.fork(num_instances = 2)
        x_dp = x_dp.slice(0, -1).collate().batch(batch_size=self.batch_size).collate()
        y_dp = y_dp.slice(1, ).collate().batch(batch_size=self.batch_size).collate()
        processed_dp = x_dp.zip(y_dp)
        return processed_dp 
    
    def _save_val_data(self, val_dataset, val_data_path):
        val_dataset.to_parquet(val_data_path)
        print(f'Validation data saved to {val_data_path}')
        return None
    
    def preprocess(self):
        split_datasets = self.dataset.train_test_split(test_size=self.test_size)

        test_dataset = split_datasets['test']
        train_dataset = split_datasets['train']

        self._save_val_data(test_dataset, self.val_data_path) if self.val_data_path is not None else None
        # source_dp = dp.iter.FileLister(path.dirname(file_path)).filter(lambda filename: filename.endswith(path.basename(file_path)))
        # raw_data = source_dp.open_files().read_from_stream().drop(0) # Drop the file name 

        train_dp = dp.iter.IterableWrapper(train_dataset).map(self._get_content)
        val_dp = dp.iter.IterableWrapper(test_dataset).map(self._get_content)

        return DataLoader2(self._causal_pipe(train_dp)), DataLoader2(self._causal_pipe(val_dp))