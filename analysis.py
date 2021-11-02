from pylab import rcParams
import copy
from sklearn.manifold import TSNE
from models.utils import importing_model
from utils.plot_images import plot_images_in_line, generate_fancy_grid
from utils.evaluation import compute_mean_variance_per_dimension
from utils.knn_on_latent import report_knn_on_latent, extract_full_data
from utils.classify_data import classify_data
from models.utils import load_model, load_coreset
from utils.plot_images import imshow
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from utils.data_loader import load_dataset, load_celeba
import torchvision
import matplotlib
matplotlib.use('agg')

from managpu import GpuManager
my_gpu = GpuManager()
using_gpu = my_gpu.set_by_memory(1)
print("Using GPU: ", using_gpu)

parser = argparse.ArgumentParser(description='ByPE-VAE')
parser.add_argument('--KNN', action='store_true', default=False,
                    help='run KNN classification on latent')
parser.add_argument('--generate', action='store_true',
                    default=False, help='generate images')
parser.add_argument('--classify', action='store_true', default=False,
                    help='train a classifier on data with augmentation')
parser.add_argument('--dir', type=str, default='',
                    help='directory of pretrained model')
parser.add_argument('--just_log_likelihood',
                    action='store_true', default=False)
parser.add_argument('--cyclic_generation', action='store_true',
                    default=False, help='cyclic generation')
parser.add_argument('--training_set_size', default=50000, type=int)
parser.add_argument('--hyper_lambda', type=float, default=0.4,
                    help='proportion of real data to augmented data')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--input_size', type=list, default=[3, 32, 32])
parser.add_argument('--count_active_dimensions',
                    action='store_true', default=False)
parser.add_argument('--grid_interpolation', action='store_true', default=False)
parser.add_argument('--tsne_visualization', action='store_true', default=False)
parser.add_argument('--hidden_units', type=int, default=1024)
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--classification_dir', type=str,
                    default='classification_report')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

print(args)

TRAIN_NUM = 50000


def plot_data(data, labels):
    k = 10
    print(data.shape)
    subplot_num = data.shape[1]
    for i in range(subplot_num):
        plt.subplot2grid((subplot_num, 1), (i, 0), colspan=1, rowspan=1)
        imshow(torchvision.utils.make_grid(data[:k, i, :].view(-1, 1, 28, 28)))
        plt.axis('off')
        print(labels[:k, i, :].squeeze())
    plt.show()


directory = args.dir


def interpolation_in_latent(model, dir, p1, p2, index=0):
    z1, _ = model.q_z(p1.to(args.device), prior=True)
    z2, _ = model.q_z(p2.to(args.device), prior=True)
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    whole_generation = []
    for radio in ratios:
        z = (1.0 - radio) * z1 + radio * z2
        image = model.generate_x_from_z(z)
        whole_generation.append(image)

    whole_generation = torch.cat(whole_generation, dim=0)
    print('whole_generation shape', whole_generation.shape)
    imshow(torchvision.utils.make_grid(
        whole_generation.reshape(-1, *model.args.input_size), nrow=11))
    save_dir = os.path.join(dir, 'grid_interpolation_coreset_select')
    os.makedirs(save_dir, exist_ok=True)
    plt.axis('off')
    plt.savefig(os.path.join(
        save_dir, 'interpolation_{}.pdf'.format(index)), bbox='tight')


def grid_interpolation_in_latent(model, dir, index, reference_image):
    z, _ = model.q_z(reference_image.to(args.device), prior=True)
    whole_generation = []
    for offset_0 in range(-2, 3, 1):
        row_generation = []
        for offset_1 in range(-2, 3, 1):
            new_z = copy.deepcopy(z)
            print(new_z.shape)
            new_z[0][0] += offset_0*3
            new_z[0][1] += offset_1*3
            image = model.generate_x_from_z(new_z, with_reparameterize=False)
            row_generation.append(image)
        whole_generation.append(torch.cat(row_generation, dim=0))
        # print("LENNN", len(whole_generation))
    whole_generation = torch.cat(whole_generation, dim=0)
    print('whole_generation shape', whole_generation.shape)
    imshow(torchvision.utils.make_grid(
        whole_generation.reshape(-1, *model.args.input_size), nrow=5))
    save_dir = os.path.join(dir, 'grid_interpolation_select')
    os.makedirs(save_dir, exist_ok=True)
    plt.axis('off')
    plt.savefig(os.path.join(
        save_dir, 'interpolation{}'.format(i)), bbox='tight')


def compute_test_metrics(test_log_likelihood, test_kl, test_re):
    test_log_likelihood.append(torch.load(
        dir + model_name + '.test_log_likelihood'))

    kl = torch.load(dir + model_name + '.test_kl')
    if type(kl) == torch.Tensor:
        kl = kl.cpu().numpy()
    test_kl.append(kl)

    reconst = torch.load(dir + model_name + '.test_re')
    if type(reconst) == torch.Tensor:
        reconst = reconst.cpu().numpy()
    test_re.append(reconst)


def cyclic_generation(start_data, dir, index):
    cyclic_generation_dir = os.path.join(dir, 'cyclic_generation')
    os.makedirs(cyclic_generation_dir, exist_ok=True)
    single_data = start_data.unsqueeze(0)
    generated_cycle = [single_data.to(args.device)]
    for i in range(29):
        single_data = \
            model.reference_based_generation_x(
                N=1, reference_image=single_data)
        generated_cycle.append(single_data)

    generated_cycle = torch.cat(generated_cycle, dim=0)
    plot_images_in_line(generated_cycle, args,
                        cyclic_generation_dir, 'cycle_{}.png'.format(index))


temp = ''
active_units_text = ''
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

for folder in sorted(os.listdir(directory)):
    print(folder)
    if os.path.isdir(directory+'/'+folder) is False:
        continue
    knn_results = []
    test_log_likelihoods, test_kl, test_reconst, active_dimensions = [], [], [], []
    knn_dictionary = {'3': [], '5': [], '7': [],
                      '9': [], '11': [], '13': [], '15': []}

    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    for filename in os.listdir(directory+'/'+folder):
        print('filename**', filename)
        dir = directory + '/' + folder+'/'+filename + '/'
        model_name_start_index = folder.find('model_name=')
        model_name = folder[model_name_start_index + len('model_name='):]
        print("MODEL NAME", model_name)

        config = torch.load(dir + model_name + '.config')
        config.device = args.device
        VAE = importing_model(config)
        model = VAE(config)
        model.to(args.device)

        if config.dataset_name == 'celeba':
            train_loader, val_loader, test_loader, config = load_celeba(config)
        else:
            train_loader, val_loader, test_loader, config = load_dataset(config,
                                                                         training_num=args.training_set_size,
                                                                         no_binarization=True)

        if args.just_log_likelihood is False:
            load_model(dir + 'checkpoint_best.pth', model)
            model.eval()
            try:
                print('prior variance', model.prior_log_variance.item())
            except:
                pass

            if args.cyclic_generation:
                with torch.no_grad():
                    for i in range(10):
                        random_image = torch.rand([784])
                        cyclic_generation(random_image, dir, index=i)

            if args.KNN:
                with torch.no_grad():
                    report_knn_on_latent(train_loader, val_loader, test_loader, model,
                                         dir, knn_dictionary, args, val=False)
            if args.generate:
                with torch.no_grad():
                    exemplars_n = 50
                    if config.model_name == 'single_conv':
                        selected_indices = torch.randint(
                            low=0, high=26000, size=(exemplars_n,))
                    else:
                        selected_indices = torch.randint(
                            low=0, high=config.training_set_size, size=(exemplars_n,))
                    reference_images, indices, labels = train_loader.dataset[selected_indices]
                    per_exemplar = 11
                    generated = model.reference_based_generation_x(
                        N=per_exemplar, reference_image=reference_images)
                    generated = generated.reshape(-1,
                                                  per_exemplar, *config.input_size)
                    rcParams['figure.figsize'] = 4, 3
                    generated_dir = dir + 'generated/'
                    if config.use_logit:
                        reference_images = model.logit_inverse(
                            reference_images)
                    generate_fancy_grid(
                        config, dir, reference_images, generated)

            if args.count_active_dimensions:
                train_loader, val_loader, test_loader, config = load_dataset(config,
                                                                             training_num=args.training_set_size,
                                                                             no_binarization=False)
                with torch.no_grad():
                    num_active = compute_mean_variance_per_dimension(
                        args, model, test_loader)
                    active_dimensions.append(num_active)

            # TODO remove loop
            if args.grid_interpolation:
                with torch.no_grad():
                    for i in range(1):
                        if config.model_name == 'single_conv':
                            if config.prior == 'CE_prior':
                                best_pts, _ = load_coreset(
                                    dir + 'checkpoint_best_coreset.pth')
                                # index_pts = torch.randint(low=0, high=config.coreset_size, size=(config.coreset_size,))
                                # image = best_pts[index_pts]
                                image = best_pts
                            else:
                                image = train_loader.dataset.tensors[0][torch.randint(low=0, high=26000,
                                                                                      size=(10,))]
                        else:
                            image = train_loader.dataset.tensors[0][torch.randint(low=0, high=args.training_set_size,
                                                                                  size=(1,))]
                        print(image.shape)
                        # for j in range(image.shape[0] - 6):
                        #     interpolation_in_latent(model, dir, image[j], image[j + 5], index=j)
                        # for j in range(image.shape[0]):
                        #     z, _ = model.q_z(image[j], prior=True)
                        #     generate_x = model.generate_x_from_z(z)
                        #     generate_x = generate_x.reshape(*model.args.input_size)
                        #     generate_x = generate_x.permute(1, 2, 0)
                        #     save_dir = os.path.join(dir, 'sample_coreset')
                        #     os.makedirs(save_dir, exist_ok=True)
                        #     plt.axis('off')
                        #
                        #     # print(generate_x.reshape(*model.args.input_size).shape)
                        #     plt.imshow(generate_x.cpu())
                        #     plt.savefig(os.path.join(save_dir, 'generate_{}.png'.format(j)), bbox='tight')
                        interpolation_in_latent(
                            model, dir, image[10], image[103], index=11)
                        interpolation_in_latent(
                            model, dir, image[107], image[110], index=12)
                        interpolation_in_latent(
                            model, dir, image[111], image[124], index=13)

                        # grid_interpolation_in_latent(model, dir, i, reference_image=image)

            if args.tsne_visualization:
                test_x, _, test_labels = extract_full_data(test_loader)
                if config.dataset_name == 'cifar10':
                    test_labels = torch.squeeze(test_labels)
                print(test_labels.max())
                print(test_labels.min())
                test_z, _ = model.q_z(test_x.to(args.device))
                print("a")
                tsne = TSNE(n_components=2)
                plt_colors = np.array(
                    ['blue', 'orange', 'green', 'red', 'cyan', 'pink', 'purple', 'brown', 'gray', 'olive'])

                points_to_visualize = tsne.fit_transform(
                    X=test_z.detach().cpu().numpy())
                print(points_to_visualize.shape)
                plt.axis('off')
                plt.scatter(points_to_visualize[:, 0], points_to_visualize[:, 1],
                            c=plt_colors[test_labels.cpu().numpy()], s=2)
                plt.savefig(dir+'tsne.pdf')
                plt.show()

            if args.classify:
                test_acc = []
                val_acc = []

                test_acc_single_run, val_acc_single_run = classify_data(train_loader, val_loader, test_loader,
                                                                        args.classification_dir, args, model)
                test_acc.append(test_acc_single_run)
                val_acc.append(val_acc_single_run)
                test_acc = np.array(test_acc)
                val_acc = np.array(val_acc)

                print('averaged test accuracy: {0:.2f} \\pm {1:.2f}'.format(
                    np.mean(test_acc), np.std(test_acc)))
                print('averaged val accuracy: {0:.2f} \\pm {1:.2f}'.format(
                    np.mean(val_acc), np.std(val_acc)))
                exit()
        else:
            compute_test_metrics(test_log_likelihoods, test_kl, test_reconst)

    if args.just_log_likelihood:
        test_log_likelihoods = np.array(test_log_likelihoods)
        print("test log-likelihood", np.mean(test_log_likelihoods),
              np.std(test_log_likelihoods))

    if args.count_active_dimensions:
        active_dimensions = np.array(active_dimensions).astype(float)
        print(np.mean(active_dimensions), np.std(active_dimensions))
