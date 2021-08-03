"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

from deoxys.experiment import Experiment
from deoxys.utils import read_file

if __name__ == '__main__':

    Experiment(
        log_base_path='../../hn_perf/unet_test'
        ).from_full_config(
            #'config/local_2d_unet_CT_W_PET.json'
            'config/my_config.json'
        ).run_experiment(
            train_history_log=True,
            model_checkpoint_period=1,
            prediction_checkpoint_period=1,
            epochs=5
    ).plot_performance().plot_prediction(
        masked_images=[i for i in range(6, 42)]
    )
