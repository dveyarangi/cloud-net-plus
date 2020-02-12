from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from cloudnet import CloudNetPlus
from jaccard_loss import FilteredJaccardLoss
import fastai
fastai.torch_core.defaults.device = 'cpu'
if __name__ == '__main__':


    path = 'Q:/FastAI/data/camvid'

    path_lbl = f'{path}/labels'
    path_img = f'{path}/images'

    get_y_fn = lambda x: f'{path_lbl}/{x.stem}_P{x.suffix}'

    codes = np.loadtxt(f'{path}/codes.txt', dtype=str); codes

    size = 192

    free = gpu_mem_get_free_no_cache()
    # the max size of bs depends on the available GPU RAM
    if free > 8200: bs=8
    else:           bs=4
    print(f"using bs={bs}, have {free}MB of GPU RAM free")

    src = (SegmentationItemList.from_folder(path_img)
           .split_by_fname_file('../valid.txt')
           .label_from_func(get_y_fn, classes=codes))

    data = (src.transform(get_transforms(), size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(imagenet_stats))

    name2id = {v:k for k,v in enumerate(codes)}
    void_code = name2id['Void']


    def acc_camvid(input, target):
        target = target.squeeze(1)
        mask = target != void_code
        return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


    metrics=acc_camvid
    # metrics=accuracy

    wd=1e-2

    model = CloudNetPlus(input_channels=3, n_classes=data.c)
#    learn = unet_learner(data, models.resnet34, pretrained=False, metrics=metrics)
    learn = Learner(data, model, metrics=metrics, path='', model_dir='models', wd=1e-2)
#    learn.loss_func = CrossEntropyFlat()
#    learn.loss_func = FlattenedLoss(FilteredJaccardLoss, axis=1)
#    learn.summary()

    lr=3e-3


    learn.fit_one_cycle(5, slice(lr), pct_start=0.9)
