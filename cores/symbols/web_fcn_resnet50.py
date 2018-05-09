import mxnet as mx
import net_symbols as syms

def create_block(data, name, num_filter, kernel, pad=0, stride=1, dilate=1, workspace=512,
                 use_global_stats=True, lr_type="alex"):
    res = syms.conv(data=data, name="res" + name, num_filter=num_filter, pad=pad, kernel=kernel, stride=stride,
                    dilate=dilate, no_bias=True, workspace=workspace, lr_type=lr_type)
    bn = syms.bn(res, name="bn" + name, use_global_stats=use_global_stats, lr_type=lr_type)
    return bn


def create_big_block(data, name, num_filter1, num_filter2, stride=1, dilate=1, pad=1, identity_map=True,
                     workspace=512, use_global_stats=True, lr_type="alex"):
    blocka = create_block(data, name=name+"_branch2a", num_filter=num_filter1, kernel=1, stride=stride,
                          workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    relu1 = syms.relu(blocka)
    blockb = create_block(relu1, name=name + "_branch2b", num_filter=num_filter1, kernel=3, dilate=dilate, pad=pad,
                          workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    relu2 = syms.relu(blockb)
    blockc = create_block(relu2, name=name+"_branch2c", num_filter=num_filter2, kernel=1, workspace=workspace,
                          use_global_stats=use_global_stats, lr_type=lr_type)
    if identity_map:
        return syms.relu(data+blockc)
    else:
        branch1 = create_block(data, name=name+"_branch1", num_filter=num_filter2, kernel=1, stride=stride,
                               workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
        return syms.relu(branch1+blockc)

def create_main(class_num, workspace=512, use_global_stats=True, lr_type="alex"):
    data = mx.symbol.Variable(name="data")

    conv1 = syms.conv(data, name="conv1", num_filter=64, pad=3, kernel=7, stride=2, workspace=workspace, lr_type=lr_type)
    bn = syms.bn(conv1, name="bn_conv1", use_global_stats=use_global_stats, lr_type=lr_type)
    relu = syms.relu(bn)
    pool1 = syms.maxpool(relu, kernel=3, stride=2, pad=1)

    res2a = create_big_block(pool1, "2a", 64, 256, identity_map=False, workspace=workspace
                             , use_global_stats=use_global_stats, lr_type=lr_type)
    res2b = create_big_block(res2a, "2b", 64, 256, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res2c = create_big_block(res2b, "2c", 64, 256, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3a = create_big_block(res2c, "3a", 128, 512, stride=2,identity_map=False, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res3b = create_big_block(res3a, "3b", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3c = create_big_block(res3b, "3c", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3d = create_big_block(res3c, "3d", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res4a = create_big_block(res3d, "4a", 256, 1024, stride=1, identity_map=False, pad=2, dilate=2,
                             workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res4b = create_big_block(res4a, "4b", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4c = create_big_block(res4b, "4c", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4d = create_big_block(res4c, "4d", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4e = create_big_block(res4d, "4e", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4f = create_big_block(res4e, "4f", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res5a = create_big_block(res4f, "5a", 512, 2048, stride=1, identity_map=False, pad=4, dilate=4,
                             workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res5b = create_big_block(res5a, "5b", 512, 2048, workspace=workspace, pad=4, dilate=4,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res5c = create_big_block(res5b, "5c", 512, 2048, workspace=workspace, pad=4, dilate=4,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    new_conv1 = syms.conv(res5c, name="new_conv1", num_filter=512, pad=12, kernel=3, dilate=12,
                          workspace=workspace, lr_type="alex10")
    new_relu1 = syms.relu(new_conv1)
    dp1 = syms.dropout(new_relu1)
    fc = syms.conv(dp1, name="fc", num_filter=512, workspace=workspace, lr_type="alex10")
    fc_relu = syms.relu(fc)
    net = syms.conv(fc_relu, name="score", num_filter=class_num, workspace=workspace, lr_type="alex10")
    return net

def create_training(class_num, workspace=512):
    net = create_main(class_num, workspace=workspace)
    softmax = mx.symbol.SoftmaxOutput(data=net, multi_output=True, use_ignore=True, ignore_label=255, name="softmax", normalization="valid")
    return softmax

def create_infer(class_num, workspace=512):
    score = create_main(class_num, workspace=workspace)
    softmax = mx.symbol.softmax(score, axis=1)
    return softmax

