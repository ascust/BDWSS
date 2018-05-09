import mxnet as mx
import net_symbols as syms


def create_aspp_block(data, dilate, name, workspace=512):
    conv1 = syms.conv(data, name="fc6_"+name, num_filter=1024, pad=dilate, kernel=3, stride=1, dilate=dilate, workspace=workspace, lr_type="alex")
    relu1 = syms.relu(conv1)
    fc = syms.conv(relu1, name="fc7_"+name, num_filter=1024, kernel=1, workspace=workspace, lr_type="alex")
    fc_relu = syms.relu(fc)
    return fc_relu


def create_convrelu_unit(data, name, num_filter, lr_type="alex", pad=1, kernel=3, stride=1, dilate=1, workspace=512):
    conv = syms.conv(data, name=name, num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, dilate=dilate, workspace=workspace, lr_type=lr_type)
    relu = syms.relu(conv)
    return relu

def create_low_feat(data, lr_type="alex", workspace=512):
    # group1
    g1_1 = create_convrelu_unit(data, "conv1_1", 64, lr_type=lr_type, workspace=workspace)
    g1_2 = create_convrelu_unit(g1_1, "conv1_2", 64, lr_type=lr_type, workspace=workspace)
    pool1 = syms.maxpool(g1_2, 3, 1, 2, pooling_convention='valid')
    # group2
    g2_1 = create_convrelu_unit(pool1, "conv2_1", 128, lr_type=lr_type, workspace=workspace)
    g2_2 = create_convrelu_unit(g2_1, "conv2_2", 128, lr_type=lr_type, workspace=workspace)
    pool2 = syms.maxpool(g2_2, 3, 1, 2, pooling_convention='valid')

    # group3
    g3_1 = create_convrelu_unit(pool2, "conv3_1", 256, lr_type=lr_type, workspace=workspace)
    g3_2 = create_convrelu_unit(g3_1, "conv3_2", 256, lr_type=lr_type, workspace=workspace)
    g3_3 = create_convrelu_unit(g3_2, "conv3_3", 256, lr_type=lr_type, workspace=workspace)
    return g3_3



def create_g_part(data, class_num, lr_type="alex", workspace=512):

    
    pool3 = syms.maxpool(data, 2, 0, 2, pooling_convention='valid')
    # group4
    g4_1 = create_convrelu_unit(pool3, "conv4_1_g", 512, lr_type=lr_type, workspace=workspace)
    g4_2 = create_convrelu_unit(g4_1, "conv4_2_g", 512, lr_type=lr_type, workspace=workspace)
    g4_3 = create_convrelu_unit(g4_2, "conv4_3_g", 512, lr_type=lr_type, workspace=workspace)
    pool4 = syms.maxpool(g4_3, 2, 0, 2, pooling_convention='valid')

    # group5
    g5_1 = create_convrelu_unit(pool4, "conv5_1_g", 512, lr_type=lr_type, workspace=workspace)
    g5_2 = create_convrelu_unit(g5_1, "conv5_2_g", 512, lr_type=lr_type, workspace=workspace)
    g5_3 = create_convrelu_unit(g5_2, "conv5_3_g", 512, lr_type=lr_type, workspace=workspace)

    pool5 = syms.maxpool(g5_3, 2, 0, 2, pooling_convention='valid')

    # group6
    conv6 = syms.conv(pool5, "fc6_g", 1024, kernel=3, lr_type=lr_type, pad=1, workspace=workspace)
    relu6 = syms.relu(conv6)

    # group7
    conv7 = syms.conv(relu6, "fc7_g", 1024, lr_type=lr_type, workspace=workspace)
    relu7 = syms.relu(conv7)
    
    pool = syms.maxpool(relu7, kernel=1, pad=0, global_pool=True)

    fc1 = syms.conv(pool, name="g_multi_fc1", num_filter=1024, workspace=workspace, lr_type="alex10")
    fc1_relu = syms.relu(fc1)
    dp1 = syms.dropout(data=fc1_relu)
    fc2 = syms.conv(dp1, name="g_multi_fc2", num_filter=1024, workspace=workspace, lr_type="alex10")
    fc2_relu = syms.relu(fc2)
    dp2 = syms.dropout(data=fc2_relu)
    g_score = syms.conv(dp2, name="g_score", num_filter=class_num, workspace=workspace, lr_type="alex10")
    return g_score


def create_seg_part(data, class_num, workspace=512, lr_type="alex"):
    

    
    pool3 = syms.maxpool(data, 3, 1, 2, pooling_convention='valid')
    # group4
    g4_1 = create_convrelu_unit(pool3, "conv4_1", 512, lr_type=lr_type, workspace=workspace)
    g4_2 = create_convrelu_unit(g4_1, "conv4_2", 512, lr_type=lr_type, workspace=workspace)
    g4_3 = create_convrelu_unit(g4_2, "conv4_3", 512, lr_type=lr_type, workspace=workspace)
    pool4 = syms.maxpool(g4_3, 3, 1, 1, pooling_convention='valid')

    # group5
    g5_1 = create_convrelu_unit(pool4, "conv5_1", 512, lr_type=lr_type, dilate=2, pad=2, workspace=workspace)
    g5_2 = create_convrelu_unit(g5_1, "conv5_2", 512, lr_type=lr_type, dilate=2, pad=2, workspace=workspace)
    g5_3 = create_convrelu_unit(g5_2, "conv5_3", 512, lr_type=lr_type, dilate=2, pad=2, workspace=workspace)

    pool5 = syms.maxpool(g5_3, 3, 1, 1, pooling_convention='valid')

    c1 = create_aspp_block(pool5, 6, "1")
    c2 = create_aspp_block(pool5, 12, "2")
    c3 = create_aspp_block(pool5, 18, "3")
    c4 = create_aspp_block(pool5, 24, "4")
    merged = c1+c2+c3+c4
    
    dp1 = syms.dropout(merged)
    fc2 = syms.conv(dp1, name="fc2", num_filter=512, workspace=workspace, lr_type="alex10")
    fc2_relu = syms.relu(fc2)
    dp2 = syms.dropout(fc2_relu)
    net = syms.conv(dp2, name="score", num_filter=class_num, workspace=workspace, lr_type="alex10")
    return net


def create_training(class_num, gweight, workspace=512):
    data = mx.symbol.Variable("data")
    g_label = mx.symbol.Variable("g_logistic_label")
    low_feat = create_low_feat(data, workspace=workspace)
    seg_part = create_seg_part(low_feat, class_num=class_num, workspace=workspace)

    g_multi = create_g_part(low_feat, class_num=class_num, workspace=workspace)
    g_multi_score = mx.symbol.LogisticRegressionOutput(data=g_multi, name="g_logistic", grad_scale=gweight, label=g_label)
    
    merge_score = mx.symbol.broadcast_plus(seg_part, g_multi)

    softmax = mx.symbol.SoftmaxOutput(data=merge_score, multi_output=True, use_ignore=True, ignore_label=255, name="softmax", normalization="valid")
    outputs = mx.symbol.Group([softmax, g_multi_score])
    return outputs

def create_infer(class_num, workspace=512):
    data = mx.symbol.Variable("data")
    low_feat = create_low_feat(data, workspace=workspace)
    seg_part = create_seg_part(low_feat, class_num=class_num, workspace=workspace)

    g_multi = create_g_part(low_feat, class_num=class_num, workspace=workspace)   
    merge_score = mx.symbol.broadcast_plus(seg_part, g_multi)

    softmax = mx.symbol.softmax(merge_score, axis=1)
    return softmax

