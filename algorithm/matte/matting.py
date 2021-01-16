import torch

from PIL import Image
from torchvision.transforms import functional as F

from . import transforms
from .model.matting_net import MattingNet


class Matting:
    def __init__(self, checkpoint_path='', gpu=False):
        torch.set_flush_denormal(True)  # flush cpu subnormal float.
        self.checkpoint_path = checkpoint_path
        self.gpu = gpu
        self.model = self.__load_model()

    def __load_model(self):
        model = MattingNet()
        if self.gpu and torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()

        # load checkpoint.
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def matting(self, image_path, with_img_trimap=False, net_img_size=-1, max_size=-1):
        """
        :param   image_path:
        :param   with_img_trimap: return origin image and pred_trimap.
        :param   net_img_size   : resize to training size for better result. (resize <= 0 => no resize)
        :param   max_size       : max size for test. (max_size <= 0 => no resize)
        :return:
                 pred_matte : shape: [H, w, 1      ] range: [0, 1]
                 image      : shape: [H, W, RGB(3) ] range: [0, 1]
                 pred_trimap: shape: [H, w, 1      ] range: [0, 1]
        """
        with torch.no_grad():
            image = self.__load_image_tensor(image_path, max_size)
            if self.gpu and torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.cpu()

            b, c, h, w = image.shape

            # resize to training size.
            if net_img_size > 0:
                resize_image = F.resize(image, [net_img_size, net_img_size], Image.BILINEAR)
                pred_matte, pred_trimap_prob, _ = self.model(resize_image)
                pred_matte = F.resize(pred_matte, [h, w])
                pred_trimap_prob = F.resize(pred_trimap_prob, [h, w], Image.BILINEAR)
            else:
                pred_matte, pred_trimap_prob, _ = self.model(image)

            pred_matte = pred_matte.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)
            image = image.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)

            pred_trimap = pred_trimap_prob.squeeze(dim=0).softmax(dim=0).argmax(dim=0)
            pred_trimap = pred_trimap.cpu().detach().unsqueeze(dim=2).numpy() / 2.

            if not with_img_trimap:
                return pred_matte

            return pred_matte, image, pred_trimap

    @staticmethod
    def __load_image_tensor(image_path, max_size=-1):
        image = Image.open(image_path).convert('RGB')
        if max_size > 0:
            [image] = transforms.ResizeIfBiggerThan(max_size)([image])
        [image] = transforms.ToTensor()([image])
        image = image.unsqueeze(dim=0)
        return image
