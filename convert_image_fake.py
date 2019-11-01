import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import sklearn.manifold
import ipdb
import cv2
import torchvision.transforms as transforms
import torchvision
import pdb
import numpy as np
import datasets
import torch
from checkpoint import Checkpointer
import attack_steps
from tqdm import tqdm
import sys
sys.path.append('../robust_classif/robustness_applications/')
import robustness.datasets
from robustness import model_utils
import robustness.attacker
class CommonModel():
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type

    def __call__(self, x):
        if self.model_type != 'robust':
            res_dict = self.model(x1=x, x2=None, class_only=True, cut_grad=False)
            lgt_glb_mlp, lgt_glb_lin = res_dict['class']
            print(res_dict['rkhs_glb'].shape)
            return res_dict['rkhs_glb']
            return lgt_glb_lin
        else:
            return self.model(x)[0]

def modify_image(images, classes, model, target_class,
                 constraint, eps, step_size, iterations, targeted, use_best,
                  do_tqdm):
    images = images.to(torch.device('cuda'))
    x = images
    intermediate_images = []
    if constraint == '2':
        if model_type == 'robust':
            step = robustness.attacker.STEPS['2'](images, eps, step_size)
        else:
            step = attack_steps.L2Step(images, eps, step_size)

    #step = attack_steps.L2Step(images, eps, step_size)

    m = 1 if targeted else -1
    nim = images.shape[0]
    intermediate_images = [x.clone().detach()]
    step_remain = iterations // 5
    steps = 1
    #criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    l2loss = torch.nn.MSELoss()
    def l2criterion(logit, classes):
        target_val = logit[:1, :].repeat(logit.shape[0], 1).detach()
        return l2loss(logit, target_val) 
    criterion = l2criterion 

    targets = torch.ones(nim).long().cuda()*target_class
    for _ in tqdm(range(iterations)):
        x = x.clone().detach().requires_grad_(True)
        class_logit = model(x)[:nim, :] 
        class_log = -criterion(class_logit, targets)
        maxim_class_loss = class_log.sum()
        print(maxim_class_loss)
        grad, = torch.autograd.grad(m*maxim_class_loss, [x])

        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
            if steps % step_remain == 0:
                intermediate_images.append(step.to_image(x))
            steps += 1

    return intermediate_images

def to_image(image_tensor):
    if model_type == 'robust':
        mean = [0,0,0]
        std =  [1,1,1]
    else:
        mean=[x/ 255.0 for x in [125.3, 123.0, 113.9]]
        std=[(x / 255.0) for x in [63.0, 62.1, 66.7]]
    mean_arr = np.array(mean)[:, None, None]
    std_arr = np.array(std)[:, None, None]
    new_im = image_tensor.cpu().numpy() 
    new_img = (new_im*std_arr) + mean_arr
    image_numpy = np.transpose(new_img, [1,2,0])*255.
    return image_numpy


def plot_modif_images(images, classes, model, target_class):
    batch_size = images.shape[0]
    images_modif = modify_image(images, classes, model, target_class, **attack_kwargs)
    images_together = torch.cat(images_modif, 0)
    im_grid = torchvision.utils.make_grid(images_together, nrow=batch_size)
    img_grid_cv2 = to_image(im_grid)
    cv2.imwrite('analysis/image_dist_L2_{}_{}.png'.format(target_class, model_type), img_grid_cv2[:, :, ::-1])

def obtain_PCA_embedding(embeddings):
    pca = PCA(n_components=2)
    pc = pca.fit_transform(embeddings)
    return pca

    
model_type = 'amdim'
if model_type == 'robust':
    attack_kwargs = {
        'constraint': '2',
        'eps': 4.0,
        'step_size': 0.5,
        'iterations': 200,
        'targeted': True,
        'use_best': False,
        'do_tqdm': True
    }
else:
    attack_kwargs = {
        'constraint': '2',
        'eps': 5.0,
        'step_size': 0.5,
        'iterations': 200,
        'targeted': True,
        'use_best': False,
        'do_tqdm': True
    }

def obtain_2d_coords(images, model, pca):
    with torch.no_grad():
        res_dict = model(x1=images, x2=images)
        global_ft = res_dict['rkhs_glb'].cpu().numpy()
        xy = pca.transform(global_ft)
    return xy

def obtain_model(model_type):
    if model_type != 'robust':
        checkpoint_path = 'runs/amdim_cpt.pth'
        checkpointer = Checkpointer()
        print('Loading model')
        model = checkpointer.restore_model_from_checkpoint(checkpoint_path)
        torch_device = torch.device('cuda')
        model = model.to(torch_device)
    else:
        dataset = robustness.datasets.CIFAR()
        model_kwargs = {
            'arch': 'resnet50',
            'dataset': dataset,
            'resume_path': f'../robust_classif/robustness_applications/models/CIFAR.pt'
        }
        model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()
    model = CommonModel(model, model_type)
    return model

def do_projection(dataset, do_pca=False):
    batch_size = 128
    num_batch_pca = 20
    counter = 0
    _, test_loader, num_classes = datasets.build_dataset(dataset, batch_size)
    all_feats = []
    classes_list = []
    out_embeddings = 'analysis/pca_embeds.npy'
    iter_dataset = iter(test_loader)
    with torch.no_grad():
        while counter < num_batch_pca:
            images, classes = next(iter_dataset)
            counter += 1
            global_ft = model(images)
            all_feats.append(global_ft.cpu().numpy())
            classes_list += [int(x) for x in classes.cpu().numpy()]
            
    all_feats = np.concatenate(all_feats, 0)
    
    if do_pca:
        pca = obtain_PCA_embedding(all_feats)
        xy = obtain_2d_coords(images, model, pca)
    else:
        xy = sklearn.manifold.TSNE().fit_transform(all_feats)

        
    return xy, classes_list

if __name__ == '__main__':
    dataset = datasets.get_dataset('C10')
    model = obtain_model(model_type) 
    perform_proj = False
    if perform_proj:
        xy, classes_proj = do_projection(dataset)
        plt.scatter(xy[:,0], xy[:,1], c=classes_proj)
        plt.savefig('analysis/tsne_img.png')
    
    pdb.set_trace() 
    # The images we will use for the experiments
    if model_type == 'robust':
        dataset = robustness.datasets.CIFAR()
        _, test_loader = dataset.make_loaders(workers=16, batch_size=10)
        test_loader.shuffle=False
    else:
        _, test_loader, num_classes = datasets.build_dataset(dataset, 10)
        test_loader.shuffle=False

    iter_dataset = iter(test_loader)
    images, classes = next(iter_dataset)
     
    # Let's look at the representations of lgt_glb_lin
    for target_class in range(10):
        plot_modif_images(images, classes, model, target_class)
        #pdb.set_trace()



