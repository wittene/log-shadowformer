
import numpy as np
import os
from torch.utils.data import DataLoader

import options
from utils import get_validation_data, apply_srgb, save_img, log_to_linear

opts = options.TestOptions(description='RGB denoising evaluation on validation set')
load_opts = opts.load_opts
output_opts = opts.output_opts
MAX_LOG_VAL = np.log(opts.log_range)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

test_dataset = get_validation_data(rgb_dir=opts.input_dir, load_opts=load_opts)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

for ii, data_test in enumerate(test_loader):
    target, input_, mask, clean_filename, noisy_filename = data_test
    print('transform')
    target = target.cpu().detach().numpy()
    if load_opts.log_transform:
        target = log_to_linear(target, log_range=load_opts.log_range)
    if load_opts.linear_transform or load_opts.log_transform:
        target = apply_srgb(target)
    print('save')
    save_img((target*255.0).astype(np.ubyte), os.path.join("results", "target_adjust", clean_filename[0]))
    print('done')
    break