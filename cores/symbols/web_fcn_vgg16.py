import mxnet as mx
import net_symbols as syms

def create_convrelu_unit(data, name, num_filter, lr_type="alex", pad=1, kernel=3, stride=1, dilate=1, workspace=512):
    conv = syms.conv(data, name=name, num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, dilate=dilate, workspace=workspace, lr_type=lr_type)
    relu = syms.relu(conv)
    return relu

def create_main(class_num, lr_type_old="alex", lr_type_new="alex", workspace=512):
    data = mx.symbol.Variable(name="data")
    # group1
    g1_1 = create_convrelu_unit(data, "conv1_1", 64, lr_type=lr_type_old, workspace=workspace)
    g1_2 = create_convrelu_unit(g1_1, "conv1_2", 64, lr_type=lr_type_old, workspace=workspace)
    pool1 = syms.maxpool(g1_2, 3, 1, 2)

    # group2
    g2_1 = create_convrelu_unit(pool1, "conv2_1", 128, lr_type=lr_type_old, workspace=workspace)
    g2_2 = create_convrelu_unit(g2_1, "conv2_2", 128, lr_type=lr_type_old, workspace=workspace)
    pool2 = syms.maxpool(g2_2, 3, 1, 2)

    # group3
    g3_1 = create_convrelu_unit(pool2, "conv3_1", 256, lr_type=lr_type_old, workspace=workspace)
    g3_2 = create_convrelu_unit(g3_1, "conv3_2", 256, lr_type=lr_type_old, workspace=workspace)
    g3_3 = create_convrelu_unit(g3_2, "conv3_3", 256, lr_type=lr_type_old, workspace=workspace)
    pool3 = syms.maxpool(g3_3, 3, 1, 2)

    # group4
    g4_1 = create_convrelu_unit(pool3, "conv4_1", 512, lr_type=lr_type_old, workspace=workspace)
    g4_2 = create_convrelu_unit(g4_1, "conv4_2", 512, lr_type=lr_type_old, workspace=workspace)
    g4_3 = create_convrelu_unit(g4_2, "conv4_3", 512, lr_type=lr_type_old, workspace=workspace)
    pool4 = syms.maxpool(g4_3, 3, 1, 1)

    # group5
    g5_1 = create_convrelu_unit(pool4, "conv5_1", 512, lr_type=lr_type_old, dilate=2, pad=2, workspace=workspace)
    g5_2 = create_convrelu_unit(g5_1, "conv5_2", 512, lr_type=lr_type_old, dilate=2, pad=2, workspace=workspace)
    g5_3 = create_convrelu_unit(g5_2, "conv5_3", 512, lr_type=lr_type_old, dilate=2, pad=2, workspace=workspace)

    pool5 = syms.maxpool(g5_3, 3, 1, 1)
    
     # group6
    conv6 = syms.conv(pool5, "fc6", 1024, kernel=3, lr_type=lr_type_old, dilate=4, pad=4, workspace=workspace)
    relu6 = syms.relu(conv6)

    # group7
    conv7 = syms.conv(relu6, "fc7", 1024, lr_type=lr_type_old, workspace=workspace)
    relu7 = syms.relu(conv7)

    new1 = syms.conv(relu7, name="new1", num_filter=512, pad=12, kernel=3, dilate=12, workspace=workspace, lr_type="alex10")
    new1_relu = syms.relu(new1)
    dp1 = syms.dropout(new1_relu)
    new2 = syms.conv(dp1, name="new2", num_filter=512, workspace=workspace, lr_type="alex10")
    new2_relu = syms.relu(new2)
    net = syms.conv(new2_relu, name="score", num_filter=class_num, workspace=workspace, lr_type="alex10")

    return net



def create_training(class_num, workspace=512):
    net = create_main(class_num, workspace=workspace)
    softmax = mx.symbol.SoftmaxOutput(data=net, multi_output=True, use_ignore=True, ignore_label=255, name="softmax", normalization="valid")
    return softmax

def create_infer(class_num, workspace=512):
    score = create_main(class_num, workspace=workspace)
    softmax = mx.symbol.softmax(score, axis=1)
    return softmax
