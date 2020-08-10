import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.util import confusion_matrix, getScores, save_images
import torch
import numpy as np
import cv2

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    save_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    model.eval()

    test_loss_iter = []
    epoch_iter = 0
    conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=np.float)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.forward()
            model.get_loss()
            epoch_iter += opt.batch_size
            gt = model.label.cpu().int().numpy()
            _, pred = torch.max(model.output.data.cpu(), 1)
            pred = pred.float().detach().int().numpy()
            save_images(save_dir, model.get_current_visuals(), model.get_image_names(), model.get_image_oriSize(), opt.prob_map)

            # Resize images to the original size for evaluation
            image_size = model.get_image_oriSize()
            oriSize = (image_size[0].item(), image_size[1].item())
            gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            conf_mat += confusion_matrix(gt, pred, dataset.dataset.num_labels)

            test_loss_iter.append(model.loss_segmentation)
            print('Epoch {0:}, iters: {1:}/{2:}, loss: {3:.3f} '.format(opt.epoch,
                                                                        epoch_iter,
                                                                        len(dataset) * opt.batch_size,
                                                                        test_loss_iter[-1]), end='\r')

        avg_test_loss = torch.mean(torch.stack(test_loss_iter))
        print ('Epoch {0:} test loss: {1:.3f} '.format(opt.epoch, avg_test_loss))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)
        print ('Epoch {0:} glob acc : {1:.3f}, pre : {2:.3f}, recall : {3:.3f}, F_score : {4:.3f}, IoU : {5:.3f}'.format(opt.epoch, globalacc, pre, recall, F_score, iou))
