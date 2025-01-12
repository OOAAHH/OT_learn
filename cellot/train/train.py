# 导入所需的Python库
from pathlib import Path

import torch
import numpy as np
import random
import pickle
from absl import logging
from absl.flags import FLAGS
from cellot import losses
from cellot.utils.loaders import load
from cellot.models.cellot import compute_loss_f, compute_loss_g, compute_w2_distance
from cellot.train.summary import Logger
from cellot.data.utils import cast_loader_to_iterator
from cellot.models.ae import compute_scgen_shift
from tqdm import trange


def load_lr_scheduler(optim, config):
    # 如果配置中没有scheduler参数,返回None
    if "scheduler" not in config:
        return None

    # 返回一个StepLR学习率调度器
    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def check_loss(*args):
    # 检查所有输入参数是否为NaN值
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_item_from_save(path, key, default):
    # 将路径转换为Path对象
    path = Path(path)
    # 如果路径不存在,返回默认值
    if not path.exists():
        return default

    # 加载checkpoint
    ckpt = torch.load(path)
    # 如果key不在checkpoint中,记录警告并返回默认值
    if key not in ckpt:
        logging.warn(f"'{key}' not found in ckpt: {str(path)}")
        return default

    # 返回checkpoint中key对应的值
    return ckpt[key]


# 最主要的修正在这里，我们要来对应新参数的引入
#def train_cellot(outdir, config):
def train_cellot(outdir, config):cell_label = None, time_label = None, GRNs_label = None):
    def state_dict(f, g, opts, **kwargs):
     '''
     这个函数返回一个字典，包含了模型 (g 和 f)、优化器 (opts.g 和 opts.f) 以及其它额外的参数（通过 kwargs）。
     它将当前模型的状态（包括参数、优化器状态等）保存下来，以便在训练过程中进行恢复（比如在断点处恢复训练）
     '''
        # 创建包含模型状态的字典
        state = {
            "g_state": g.state_dict(),
            "f_state": f.state_dict(),
            "opt_g_state": opts.g.state_dict(),
            "opt_f_state": opts.f.state_dict(),
        }
        # 更新额外的参数
        state.update(kwargs)

        return state

    def evaluate():
        '''
        这个函数用于评估当前模型的性能。它从测试集获取 source 和 target 数据，
        计算模型的生成器损失 (gl)、判别器损失 (fl)、W2 距离、最大均值差异 (mmd) 等指标。
        通过 logger.log() 记录评估结果。
        '''
        # 获取测试数据
        target = next(iterator_test_target)
        source = next(iterator_test_source)
        source.requires_grad_(True)
        # 计算transport
        transport = g.transport(source)

        transport = transport.detach()
        with torch.no_grad():
            # 计算各种损失值
            gl = compute_loss_g(f, g, source, transport).mean()
            fl = compute_loss_f(f, g, source, target, transport).mean()
            dist = compute_w2_distance(f, g, source, target, transport)
            mmd = losses.compute_scalar_mmd(
                target.detach().numpy(), transport.detach().numpy()
            )

        # 记录评估结果
        logger.log(
            "eval",
            gloss=gl.item(),
            floss=fl.item(),
            jloss=dist.item(),
            mmd=mmd,
            step=step,
        )
        # 检查损失值是否为NaN
        check_loss(gl, gl, dist)

        return mmd

    # 创建日志记录器
    '''
    load 函数从配置中加载模型（f 和 g）以及优化器（opts）。如果有已保存的检查点，它会从指定路径（cachedir / "last.pt"）恢复模型状态
    确保你可以恢复之前的训练状态，避免从头开始训练。此外，loader 是数据加载器，负责加载数据集。
    '''
    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    # 加载模型、优化器和数据加载器
    (f, g), opts, loader = load(config, restore=cachedir / "last.pt")
    iterator = cast_loader_to_iterator(loader, cycle_all=True)

    # 获取训练迭代次数
    n_iters = config.training.n_iters
    # 加载当前步数
    step = load_item_from_save(cachedir / "last.pt", "step", 0)

    # 加载最小MMD值
    minmmd = load_item_from_save(cachedir / "model.pt", "minmmd", np.inf)
    mmd = minmmd

    # 检查是否需要配对批处理
    if 'pair_batch_on' in config.training:
        keys = list(iterator.train.target.keys())
        test_keys = list(iterator.test.target.keys())
    else:
        keys = None

    # 创建进度条
    ticker = trange(step, n_iters, initial=step, total=n_iters)
    # 开始训练循环
    for step in ticker:
        # 如果使用配对批处理
        if 'pair_batch_on' in config.training:
            assert keys is not None
            # 随机选择一个key
            key = random.choice(keys)
            iterator_train_target = iterator.train.target[key]
            iterator_train_source = iterator.train.source[key]
            try:
                iterator_test_target = iterator.test.target[key]
                iterator_test_source = iterator.test.source[key]
            # 处理IID模式下的OOD设置
            except KeyError:
                test_key = random.choice(test_keys)
                iterator_test_target = iterator.test.target[test_key]
                iterator_test_source = iterator.test.source[test_key]

        else:
            # 不使用配对批处理时的迭代器设置
            iterator_train_target = iterator.train.target
            iterator_train_source = iterator.train.source
            iterator_test_target = iterator.test.target
            iterator_test_source = iterator.test.source

        # 获取目标数据
        target = next(iterator_train_target)
        # 内部训练循环
        for _ in range(config.training.n_inner_iters):
            '''
            这是 CellOT 模型的核心训练循环。每一个训练步骤都会获取新的训练数据（source 和 target），
            并使用反向传播更新模型参数。
            '''
            # 获取源数据并设置梯度
            source = next(iterator_train_source).requires_grad_(True)

            # 训练生成器
            opts.g.zero_grad()
            gl = compute_loss_g(f, g, source).mean()
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()

            gl.backward()
            opts.g.step()

        # 获取新的源数据
        source = next(iterator_train_source).requires_grad_(True)

        # 训练判别器
        opts.f.zero_grad()
        fl = compute_loss_f(f, g, source, target).mean()
        fl.backward()
        opts.f.step()
        # 检查损失值
        check_loss(gl, fl)
        f.clamp_w()

        # 定期记录训练日志
        if step % config.training.logs_freq == 0:
            logger.log("train", gloss=gl.item(), floss=fl.item(), step=step)

        # 定期评估模型
        if step % config.training.eval_freq == 0:
            mmd = evaluate()
            if mmd < minmmd:
                minmmd = mmd
                # 保存最佳模型
                torch.save(
                    state_dict(f, g, opts, step=step, minmmd=minmmd),
                    cachedir / "model.pt",
                )

        # 定期保存检查点
        if step % config.training.cache_freq == 0:
            torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")
            logger.flush()

    # 保存最终模型
    torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")
    logger.flush()

    return


def train_auto_encoder(outdir, config):
    def state_dict(model, optim, **kwargs):
        # 创建包含模型状态的字典
        state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }

        # 如果模型有code_means属性,添加到状态字典
        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means

        # 更新额外的参数
        state.update(kwargs)

        return state

    def evaluate(vinputs):
        # 评估模型性能
        with torch.no_grad():
            loss, comps, _ = model(vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)
        return loss

    # 创建日志记录器
    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    # 加载模型、优化器和数据加载器
    model, optim, loader = load(config, restore=cachedir / "last.pt")

    # 创建数据迭代器
    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    # 加载学习率调度器
    scheduler = load_lr_scheduler(optim, config)

    # 获取训练迭代次数
    n_iters = config.training.n_iters
    # 加载当前步数
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    # 加载最佳评估损失
    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf
    )

    eval_loss = best_eval_loss

    # 创建进度条
    ticker = trange(step, n_iters, initial=step, total=n_iters)
    # 开始训练循环
    for step in ticker:

        # 设置为训练模式
        model.train()
        # 获取训练数据
        inputs = next(iterator.train)
        # 训练步骤
        optim.zero_grad()
        loss, comps, _ = model(inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        # 定期记录训练日志
        if step % config.training.logs_freq == 0:
            logger.log("train", loss=loss.item(), step=step, **comps)

        # 定期评估模型
        if step % config.training.eval_freq == 0:
            model.eval()
            eval_loss = evaluate(next(iterator.test))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)
                # 保存最佳模型
                torch.save(sd, cachedir / "model.pt")

        # 定期保存检查点
        if step % config.training.cache_freq == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")
            logger.flush()

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

    # 如果是scgen模型且需要计算shift
    if config.model.name == "scgen" and config.get("compute_scgen_shift", True):
        labels = loader.train.dataset.adata.obs[config.data.condition]
        compute_scgen_shift(model, loader.train.dataset, labels=labels)

    # 保存最终模型
    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")
    logger.flush()


def train_popalign(outdir, config):
    def evaluate(config, data, model):
        # 获取控制组和处理组的测试数据索引
        idx_control_test = np.where(data.obs[
            config.data.condition] == config.data.source)[0]
        idx_treated_test = np.where(data.obs[
            config.data.condition] == config.data.target)[0]

        # 计算预测值和目标值
        predicted = transport_popalign(model, data[idx_control_test].X)
        target = np.array(data[idx_treated_test].X)

        # 计算性能指标
        mmd = losses.compute_scalar_mmd(target, predicted)
        wst = losses.wasserstein_loss(target, predicted)

        # 记录评估结果
        logger.log(
            "eval",
            mmd=mmd,
            wst=wst,
            step=1
        )

    # 创建日志记录器
    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"

    # 加载数据集和之前的模型参数
    model, _, dataset = load(config, restore=cachedir / "last.pt",
                             return_as="dataset")
    train_data = dataset["train"].adata
    test_data = dataset["test"].adata

    # 如果模型缺少必要的组件
    if not all(k in model for k in ("dim_red", "gmm_control", "response")):

        # 根据嵌入类型选择降维方法
        if config.model.embedding == 'onmf':
            q, nfeats, errors = onmf(train_data.X.T)
            W, proj = choose_featureset(
                train_data.X.T, errors, q, nfeats, alpha=3, multiplier=3)

        else:
            W = np.eye(train_data.X.shape[1])
            proj = train_data.X

        # 获取控制组和处理组的训练数据索引
        idx_control_train = np.where(train_data.obs[
            config.data.condition] == config.data.source)[0]
        idx_treated_train = np.where(train_data.obs[
            config.data.condition] == config.data.target)[0]

        # 为控制组和处理组构建高斯混合模型
        gmm_control = build_gmm(
            train_data.X[idx_control_train, :].T,
            proj[idx_control_train], ks=(3), niters=2,
            training=.8, criteria='aic')
        gmm_treated = build_gmm(
            train_data.X[idx_treated_train, :].T,
            proj[idx_treated_train], ks=(3), niters=2,
            training=.8, criteria='aic')

        # 对齐两个混合模型的组件
        align, _ = align_components(gmm_control, gmm_treated, method="ref2test")

        # 计算每个控制组件的扰动响应
        res = get_perturbation_response(align, gmm_control, gmm_treated)

        # 保存所有结果到状态字典
        model = {"dim_red": W,
                 "gmm_control": gmm_control,
                 "gmm_treated": gmm_treated,
                 "response": res}
        state_dict = model
        pickle.dump(state_dict, open(cachedir / "last.pt", 'wb'))
        pickle.dump(state_dict, open(cachedir / "model.pt", 'wb'))

    else:
        # 如果模型已有所需组件,直接加载
        W = model["dim_red"]
        gmm_control = model["gmm_control"]
        gmm_treated = model["gmm_treated"]
        res = model["response"]

    # 在测试集上评估性能
    evaluate(config, test_data, model)
