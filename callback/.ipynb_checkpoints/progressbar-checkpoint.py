import time
from tqdm import tqdm


class ProgressBar(object):

    def __init__(self, n_total,width=30, disable=False):
        self.width = width
        self.n_total = n_total
        self.tqdm = tqdm(total=self.n_total,dynamic_ncols=True, disable=disable)
        self.start_time = time.time()

    def batch_step(self, step, info, bar_type='Training'):
        self.tqdm.set_description(f"[{bar_type}]")
        self.tqdm.update(1)
        if len(info) != 0:
            info_str = ' - '.join([f'{key}: {value:.4f}' for key, value in info.items()])
            tqdm.write(f'Step {step + 1}/{self.n_total} - {info_str}')

    def close(self):
        # close the progressbar after utilization.
        self.tqdm.close()