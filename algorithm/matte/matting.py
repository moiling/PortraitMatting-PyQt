import torch

from PIL import Image
from torchvision.transforms import functional as F

from . import transforms


class Matting:
    def __init__(self, model_path='', model_fix_path='', gpu=False):
        torch.set_flush_denormal(True)  # flush cpu subnormal float.
        self.model_path = model_path
        self.model_fix_path = model_fix_path
        self.gpu = gpu
        self.model, self.model_fix = self.__load_model()

    def __load_model(self):
        # model = MattingNet()
        model = torch.jit.load(self.model_path, map_location='cpu')
        model_fix = torch.jit.load(self.model_fix_path, map_location='cpu')
        if self.gpu and torch.cuda.is_available():
            model.cuda()
            model_fix.cuda()
        else:
            model.cpu()
            model_fix.cpu()

        # load checkpoint.
        # checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_fix.eval()
        return model, model_fix

    def matting(self, image_path, with_img_trimap=False, net_img_size=-1, max_size=-1, trimap=None):
        """
        :param   trimap:
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
            trimap_3 = self.__load_trimap_tensor(trimap, max_size)
            if self.gpu and torch.cuda.is_available():
                image = image.cuda()
                if trimap_3 is not None:
                    trimap_3 = trimap_3.cuda()
            else:
                image = image.cpu()
                if trimap_3 is not None:
                    trimap_3 = trimap_3.cpu()

            b, c, h, w = image.shape

            # resize to training size.
            if net_img_size > 0:
                resize_image = F.resize(image, [net_img_size, net_img_size], Image.BILINEAR)

                if trimap_3 is not None:
                    resize_trimap = F.resize(trimap_3, [net_img_size, net_img_size], Image.BILINEAR)
                    pred_matte, pred_trimap_prob, _ = self.model_fix(resize_image, resize_trimap)
                else:
                    pred_matte, pred_trimap_prob, _ = self.model(resize_image)

                pred_matte = F.resize(pred_matte, [h, w])
                pred_trimap_prob = F.resize(pred_trimap_prob, [h, w], Image.BILINEAR)

            else:
                if trimap_3 is not None:
                    pred_matte, pred_trimap_prob, _ = self.model_fix(image, trimap_3)
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

    def __load_trimap_tensor(self, trimap, max_size=-1):
        if trimap is None:
            return None
        # trimap = Image.open(trimap_path).convert('L')
        trimap = Image.fromarray(trimap).convert('L')

        if max_size > 0:
            [trimap] = transforms.ResizeIfBiggerThan(max_size)([trimap])
        [trimap] = transforms.ToTensor()([trimap])

        # get 3-channels trimap.
        trimap_3 = trimap.repeat(3, 1, 1)
        trimap_3[0, :, :] = (trimap_3[0, :, :] <= 0.1).float()
        trimap_3[1, :, :] = ((trimap_3[1, :, :] < 0.9) & (trimap_3[1, :, :] > 0.1)).float()
        trimap_3[2, :, :] = (trimap_3[2, :, :] >= 0.9).float()

        trimap_3 = trimap_3.unsqueeze(dim=0)
        return trimap_3
