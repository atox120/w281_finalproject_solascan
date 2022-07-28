import copy
import math
import random
import time

import numpy as np
# import the necessary packages
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from app.utils import ImageWrapper


class Classifier:
    def __init__(self, defect, not_defect, model_class, data_class):

        # These are the images
        self.defect = defect
        self.not_defect = not_defect
        self.model_class = model_class
        self.data_class = data_class
        self.model = None
        self.pca = None

        self.transformation_label = ''
        self.pre_transform = None
        self.post_transform = None
        self.train_data = ()
        self.cv_data = ()
        self.reshape_pre = None
        self.reshape_pst = None
        self.image_labels = []

    def _process_data(self, params):
        """
        Apply the data object to the images
        """

        if self.data_class is not None:

            # Create an object of the data class
            self.data_obj = self.data_class(consume_kwargs=False, **params)

            # Check the ones that were used and remove them from the list
            keys = list(params.keys())
            for key in keys:
                if key in self.data_obj.__dict__ or key in self.data_obj.params:
                    del (params[key])

            # Process the  data through the data object
            defect = self.data_obj << self.defect
            not_defect = self.data_obj << self.not_defect

            defect = defect[-1]
            not_defect = not_defect[-1]

            self.reshape_pst = (~not_defect).shape[1:]

            # The first items is the defect class type, remove it?
            self.transformation_label = defect.category
        else:
            defect = self.defect
            not_defect = self.not_defect
            self.reshape_pst = self.reshape_pre

        return defect, not_defect

    def _apply_pca(self, pca_dims, x_train, x_cv):
        """
        Get the desired number of dimensions from the data
        """

        if pca_dims is not None:
            self.pca = PCA(n_components=pca_dims)

            # Transform the train data
            x_train = self.pca.fit_transform(x_train)

            # Transform the CV data
            x_cv = self.pca.transform(x_cv)
        return x_train, x_cv

    def misclassified(self):
        """
        Creates a confusion matrix and an ImageWrapper object of misclassified images

        :return confusion matrix, tuple of misclassified images
        """

        out_array = []
        confusion = np.zeros((2, 2))
        if self.cv_data:
            # noinspection PyTupleAssignmentBalance
            x_cv, y_true, y_pred, cv_indices = self.cv_data

            for yt in np.unique(y_true):
                for yp in np.unique(y_pred):

                    # These are the mismatched predictions
                    bad_pred = np.logical_and(y_true == yt, y_pred == yp)
                    confusion[yt, yp] = np.sum(bad_pred)

                    if yt == yp:
                        continue
                    # bad_pred = np.logical_and(bad_pred, y_true != y_pred)

                    # These are the image lables for the bad predictions
                    image_labels = self.image_labels[cv_indices[bad_pred].astype(int)]

                    # Reformat it into the image
                    x_mis_pre = self.pre_transform[cv_indices[bad_pred].astype(int), :]
                    x_mis_post = self.post_transform[cv_indices[bad_pred].astype(int), :]

                    x_mis_pre = x_mis_pre.reshape(np.sum(bad_pred), *self.reshape_pre)
                    x_mis_post = x_mis_post.reshape(np.sum(bad_pred), *self.reshape_pst)

                    if not x_mis_pre.size == 0:
                        imw_pre = ImageWrapper(x_mis_pre, image_labels=image_labels.tolist(),
                                               category=f'misclassified True: {yt} Pred: {yp}')
                        imw_post = ImageWrapper(
                            x_mis_post, image_labels=image_labels.tolist(),
                            category=self.transformation_label + f'\nmisclassified True: {yt} Pred: {yp}')

                        out_array.append((imw_pre, imw_post))

        return confusion, out_array

    def fit(self, **params):
        """
        1. Apply the params to the data_class and model_class a
        2. Apply PCA
        3. Train the model
        4. Test the model against Cross Validation data
        """

        try:
            pca_dims = params['pca_dims']
            del params['pca_dims']
        except KeyError:
            pca_dims = None

        try:
            seed = params['seed']
            del params['seed']
        except KeyError:
            seed = random.randint(0, 2 ** 32)

        # Extract the images before transformation

        # Get the data as numpy array
        pre_defect = ~self.defect
        pre_not_defect = ~self.not_defect
        # These are the values to reshape the images back to
        self.reshape_pre = pre_defect.shape[1:]

        x_vals = np.vstack((pre_defect, pre_not_defect))
        self.pre_transform = x_vals

        # This is where the Data object class is applied
        defect, not_defect = self._process_data(params)

        # The remaining parameters go into the model class
        self.model = self.model_class(**params)
        self.image_labels = np.array(defect.image_labels + not_defect.image_labels)

        # Get the data as numpy array
        defect = ~defect
        not_defect = ~not_defect

        # Flatten the image and convert into X array
        defect = defect.reshape((defect.shape[0], -1))
        not_defect = not_defect.reshape((not_defect.shape[0], -1))
        x_vals = np.vstack((defect, not_defect))
        self.post_transform = x_vals

        # Add indices for referencing later
        x_vals = np.hstack((np.arange(x_vals.shape[0])[:, np.newaxis], x_vals))

        # Flatten the image and convert into y array
        y_defect = np.ones((defect.shape[0],)).astype(int)
        y_not_defect = np.zeros((not_defect.shape[0],)).astype(int)
        y_vals = np.concatenate((y_defect, y_not_defect))

        # Split into train and cv set
        x_train, x_cv, y_train, y_cv = \
            train_test_split(x_vals, y_vals, test_size=0.20, random_state=seed)

        # Apply the PCA to the train set and transform the cv set too
        cv_indices = x_cv[:, 0]
        x_train, x_cv = self._apply_pca(pca_dims, x_train[:, 1:], x_cv[:, 1:])

        # Fit the model for the train data
        self.model.fit(x_train, y_train)

        # Predict the model on the cv
        y_pred = self.model.predict(x_cv)

        # Save the CV indices to refernce later
        self.cv_data = (x_cv, y_cv, y_pred, cv_indices)

        # Calculate the balanced accuracy score
        score = balanced_accuracy_score(y_cv, y_pred)

        return score


def get_output_shape(model, image_dim):
    return model(torch.rand(*image_dim)).data.shape


class CNN(nn.Module):
    def __init__(self, num_output_classes, channels=((1, 5), (20, 3), (20, 3)),
                 input_shape=(1, 1, 174, 174)):
        # call the parent constructor
        super(CNN, self).__init__()

        self.layers = nn.ModuleList([])
        in_channel = channels[0][0]
        output_shape = input_shape
        for out_channel, kernel_size in channels[1:]:
            # Initialize the number of layers
            self.layers.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                         kernel_size=(kernel_size, kernel_size)))
            output_shape = get_output_shape(self.layers[-1], output_shape)

            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(num_features=in_channel[1]))
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            output_shape = get_output_shape(self.layers[-1], output_shape)
            in_channel = out_channel

        in_layers = int(np.prod(output_shape))
        out_layers = math.ceil(in_layers / 10)

        # Add a flatten layer
        self.layers.append(nn.Flatten())

        # Two layers of output
        self.layers.append(nn.Linear(in_features=in_layers, out_features=out_layers))
        self.layers.append(nn.ReLU())

        # initialize our softmax classifier
        self.layers.append(nn.Linear(in_features=out_layers, out_features=num_output_classes))
        self.layers.append(nn.LogSoftmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


# noinspection PyPep8Naming
class DataLoader:
    def __init__(self, X, y, batchsize=1024, shuffleidx=True):
        self.X, self.y = X, y
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batchsize = batchsize
        if shuffleidx:
            index = list(range(X.shape[0]))
            random.shuffle(index)

            # This is the reshuffled data
            self.X = X[index]
            self.y = y[index]

    def __len__(self):
        return math.ceil(self.X.shape[0]/self.batchsize)

    def __getitem__(self, idx):

        # Get the next ordered batch of X and Y
        batchidx = slice(idx*self.batchsize, (idx+1)*self.batchsize)
        x = self.X[batchidx]
        y = self.y[batchidx]

        # send it to the device
        x = torch.tensor(x).float().to(self.device)
        y = torch.tensor(y).long().to(self.device)

        return x, y


# noinspection PyPep8Naming
class ModelCNN:
    def __init__(self, defect, not_defect, model_params, optimizer_params, scheduler_params):

        # Format the data for the data loader
        defect = self._format(defect)
        not_defect = self._format(not_defect)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batchsize = 32
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        # Stack the X's together
        self.X = np.concatenate((defect, not_defect), axis=0)
        self.y = np.concatenate((np.ones((defect.shape[0], )),
                                 np.zeros((not_defect.shape[0], ))), axis=0)

    @staticmethod
    def _format(inm):
        # If its a tuple of values then get the last one
        if isinstance(inm, tuple):
            inm = inm[-1]

        if isinstance(inm, ImageWrapper):
            inm = ~inm

        # If there are three dimensions in the data then add one for channel
        if len(inm.shape) == 3:
            inm = inm[:, np.newaxis, :, :]

        return inm

    def _get_optimizer(self):
        """

        :return:
        """
        model = self.model
        optimizer_params = self.optimizer_params

        optimizer_name = optimizer_params['name']
        other_params = copy.deepcopy(optimizer_params)
        del other_params['name']
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **other_params)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), **other_params)
        else:
            raise KeyError(f'Currently unsupported optimizer {optimizer_name}')

        self.optimizer = optimizer

        return optimizer

    def _get_scheduler(self, n_steps, num_epochs):
        """

        :return:
        """

        t_mul = self.scheduler_params['t_mul']
        lr_min = self.scheduler_params['lr_min']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=n_steps * num_epochs + 1,
            T_mult=t_mul,
            eta_min=lr_min
        )
        self.scheduler = scheduler

        return scheduler

    @staticmethod
    def _train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch):
        """

        :param model:
        :param train_loader:
        :param criterion:
        :param optimizer:
        :param scheduler:
        :param epoch:
        :return:
        """
        num_steps = len(train_loader)

        model.train()

        start = time.perf_counter()
        losses = []
        for i in range(num_steps):
            # Set the X and Y data
            # They should be on the device now
            x_data, y_data = train_loader[i]

            # Do I need to send to cuda?

            # Forward pass the model
            pred = model(x_data)

            # Calculate loss
            loss = criterion(pred, y_data)

            # Zero out the gradients as backward wil accumulate
            # the gradient from the previous step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step(num_steps*epoch+i)

            # Append the loss to the list
            losses.append(loss.item())

            # print(f'\t\t Epoch {epoch} iteration {i} train loss {losses[-1]}')

        # print(f'\t Epoch {epoch} took {time.perf_counter()-start}s ')

        return losses

    @staticmethod
    def _validate_one_epoch(model, val_loader, criterion, epoch):

        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(len(val_loader)):
                # Load the validation data
                x_data, y_data = val_loader[i]

                # Make a prediction using the model
                pred = model(x_data)

                # Calculate the loss from the model
                loss = criterion(pred, y_data)

                # Append the losses
                losses.append(loss.item())

                # print(f'\t\t Epoch {epoch} iteration {i} validation loss {losses[-1]}')

        return losses

    def _fit(self, train_loader, val_loader, num_epochs, criterion):
        """

        :param train_loader:
        :param val_loader:
        :param num_epochs:
        :param criterion:
        :return:
        """

        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        iteration_loss = []
        epoch_train_loss = []
        epoch_val_loss = []
        for epoch in range(num_epochs):
            # Send the model to CUDA or CPU based on GPU availability
            model.to(self.device)

            # Train one epoch
            it_loss = self._train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
            iteration_loss += it_loss
            epoch_train_loss.append(np.mean(it_loss))

            # Look at validation loss from this epoch
            it_loss = self._validate_one_epoch(model, val_loader, criterion, epoch)
            epoch_val_loss.append(np.mean(it_loss))

            # Epoch train and validation losses
            print(f'Epoch {epoch} train loss {epoch_train_loss[-1]} val loss {epoch_val_loss[-1]}')

            # If it is the best loss till now then save the model
            state_dict = copy.deepcopy(self.model_params)
            state_dict.update()
            if len(epoch_val_loss) > 1 and epoch_val_loss[-1] < np.min(epoch_val_loss[:-1]):
                state = {'epoch': epoch + 1, 'arch': 'CNN', 'best_loss': epoch_train_loss[-1],
                         'state_dict': model.state_dict(), 'model_params': self.model_params}
                torch.save(state, f'../models/cnn_model_best.pth')

        # Load the final model
        state_dict = torch.load(open(f'../models/cnn_model_best.pth', 'rb'))
        model = CNN(**state_dict['model_params'])

        model.load_state_dict(state_dict['state_dict'])
        model.to(self.device)

        return model

    def predict(self, x_data):
        """

        :param x_data: Make prediction for x
        :return:
        """

        model = self.model
        model.eval()
        with torch.no_grad():
            x_data = torch.tensor(x_data).float().to(self.device)
            pred = model(x_data)
            pred = pred.argmax(axis=1)
            if "cuda" in self.device:
                pred = pred.cpu().detach().numpy()

        return pred

    def fit(self, num_epochs=10, seed=None):
        """

        :return:
        """

        model_params = copy.deepcopy(self.model_params)

        # During model creation a random sample is generated
        # for finding size of layers. Make the first index
        # 1 so that a massive array is not created
        model_params['input_shape'] = list(self.X.shape)
        model_params['input_shape'][0] = 1
        # Update the input model params
        self.model_params = model_params
        self.model = CNN(**model_params)

        self._get_optimizer()

        seed = random.randint(0, 2**32) if seed is None else seed

        # Split into train and test data
        x_train, x_cv, y_train, y_cv = \
            train_test_split(self.X, self.y, test_size=0.20, random_state=seed)

        # Create a data loader for train and test data
        train_loader = DataLoader(x_train, y_train, batchsize=self.batchsize)
        val_loader = DataLoader(x_cv, y_cv, batchsize=self.batchsize)

        # Create a scheduler object
        n_steps = len(train_loader)

        # This schedules the learning rate
        self._get_scheduler(n_steps, num_epochs)

        # Loss criterion
        criterion = nn.CrossEntropyLoss()

        self.model = self._fit(train_loader, val_loader, num_epochs, criterion)

        # Calculate model accuracy
        y_pred = self.predict(x_cv)

        # Balanced accuracy score
        score = balanced_accuracy_score(y_cv, y_pred)

        return score


if __name__ == "__main__":
    # Defects have a  mean of 3
    defec = np.random.randn(400, 174, 174) + 1
    # No defects have a mean of 0
    no_defec = np.random.randn(400, 174, 174)

    op = {'name': 'sgd', 'lr': 0.001, 'nesterov': True, 'momentum': 0.9}
    sp = {'lr_min': 0.0001, 't_mul': 2}
    mp = {'num_output_classes': 2, 'channels': (1, 20, 20)}

    # Fit this model
    mod = ModelCNN(defec, no_defec, mp, op, sp)
    sc = mod.fit()
    print(sc)


