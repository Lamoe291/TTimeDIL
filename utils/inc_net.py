import copy
import logging
import os
import torch
from torch import nn
from torch.nn import functional as F
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, CosineLinearFeature, SimpleContinualLinear
from backbone.prompt import CodaPrompt
import timm
import math
from backbone.lora import LoRA_ViT_timm, TaskWeightedPEFT_ViT_timm
from backbone.lora_efficient import LoRA_ViT_timm as LoRA_ViT_timm_efficient
from backbone.lora_finetune import LoRA_ViT_timm as Ft_LoRA_ViT_timm



def _resolve_vit_backbone_name(backbone_name):
    normalized_name = backbone_name.lower()
    if "clip" in normalized_name:
        return "vit_base_patch16_clip_224.openai"
    elif "dinov3" in normalized_name:
        return "vit_base_patch16_dinov3.lvd1689m"
    elif "dinov2" in normalized_name:
        return "vit_base_patch16_224_dinov2"
    elif "dino" in normalized_name:
        return "vit_base_patch16_224_dino"
    elif normalized_name.startswith("pretrained_vit_b16_224"):
        return "vit_base_patch16_224"
    return normalized_name


def _create_pretrained_vit_model(backbone_name, pretrained=False, num_classes=0):
    base_model_name = _resolve_vit_backbone_name(backbone_name)
    if "clip" in base_model_name:
        candidate_names = [base_model_name, "vit_base_patch16_224_clip_laion2b", "vit_base_patch16_224"]
    elif "dinov3" in base_model_name:
        candidate_names = [base_model_name, "vit_base_patch16_224_dinov2", "vit_base_patch16_224_dino", "vit_base_patch16_224"]
    elif "dinov2" in base_model_name or "dino" in base_model_name:
        candidate_names = [base_model_name, "vit_base_patch16_224_dino", "vit_base_patch16_224"]
    else:
        candidate_names = [base_model_name]

    last_error = None
    for candidate_name in candidate_names:
        try:
            model = timm.create_model(candidate_name, pretrained=pretrained, num_classes=num_classes)
            setattr(model, "_selected_backbone_name", candidate_name)
            print(f"[backbone] using pretrained ViT backbone '{candidate_name}' (requested '{backbone_name}')")
            return model, candidate_name
        except Exception as exc:
            last_error = exc
            print(f"[backbone] failed to create pretrained ViT backbone '{candidate_name}' (requested '{backbone_name}'): {exc}")

    if last_error is not None:
        raise last_error
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    setattr(model, "_selected_backbone_name", "vit_base_patch16_224")
    print(f"[backbone] using pretrained ViT backbone 'vit_base_patch16_224' (requested '{backbone_name}')")
    return model, "vit_base_patch16_224"


def _load_backbone_checkpoint(model, args):
    checkpoint_path = args.get("backbone_checkpoint") or args.get("pretrained_weights_path") or args.get("checkpoint_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("model")
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    if not isinstance(state_dict, dict):
        return model

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1)
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Loaded pretrained checkpoint from {checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})")
    return model


def get_backbone(args, pretrained=False, num_tasks=1, current_task=0):
    name = args["backbone_type"].lower()
    ## Lora version
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224" or name == "vit_base_patch16_224_dino" or name == "vit_base_patch16_224_dinov2" or name == "vit_base_patch16_dinov3.lvd1689m" or name == "vit_base_patch16_224_dinov3" or name == "vit_base_patch16_clip_224.openai" or name == "vit_base_patch16_224_clip" or name == "pretrained_vit_b16_224_clip" or name == "pretrained_vit_b16_224_dinov3":
        if args["model_name"] == "finetune":
            model, selected_backbone_name = _create_pretrained_vit_model(name, pretrained=pretrained, num_classes=0)
            model = _load_backbone_checkpoint(model, args)
            model = Ft_LoRA_ViT_timm(vit_model=model.eval(), r=args.get('lora_rank',10),  num_classes=10, increment=args['increment'], filepath=args['filepath'] + args['prefix'] + '/' , reinit = args.get('joined',False), use_lora = args.get('use_lora',True), update_backbone = args.get('update_backbone',False))
        elif args["model_name"] == "ttime-peft":
            model, selected_backbone_name = _create_pretrained_vit_model(name, pretrained=pretrained, num_classes=0)
            model = _load_backbone_checkpoint(model, args)
            model = TaskWeightedPEFT_ViT_timm(vit_model=model.eval(), use_adapter=args.get('use_adapter', False), adapter_bottleneck=args.get('adapter_bottleneck', 64), adapter_dropout=args.get('adapter_dropout', 0.0), adapter_scale=args.get('adapter_scale', 1.0), use_prompt=args.get('use_prompt', False), prompt_length=args.get('prompt_length', 5), num_tasks=num_tasks, current_task=current_task, top_k=args.get('top_k', 3), freeze_backbone=args.get('freeze_backbone', True), init_from_previous_task=args.get('init_from_previous_task', True), use_prefix=args.get('use_prefix', False), prefix_length=args.get('prefix_length', 5))
        elif args["model_name"] == "ttime" :
            model, selected_backbone_name = _create_pretrained_vit_model(name, pretrained=pretrained, num_classes=0)
            model = _load_backbone_checkpoint(model, args)
            if args.get('efficient_inference',False):
                model = LoRA_ViT_timm_efficient(vit_model=model.eval(), r=args.get('lora_rank',10),  num_classes=10, increment=args['increment'], filepath=args['filepath'] + args['prefix'] + '/' , cur_task_index=0, top_k=args.get('top_k',3))
            else:
                model = LoRA_ViT_timm(vit_model=model.eval(), r=args.get('lora_rank',10),  num_classes=10, increment=args['increment'], filepath=args['filepath'] + args['prefix'] + '/' )

        setattr(model, '_selected_backbone_name', selected_backbone_name)
        setattr(model, '_requested_backbone_name', name)
        print(f"[backbone] final backbone wrapper={type(model).__name__}, base_backbone='{getattr(model, '_selected_backbone_name', selected_backbone_name)}', requested_backbone='{getattr(model, '_requested_backbone_name', name)}'")
        # model = nn.DataParallel(model)
        model.out_dim = 768
        return model

    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    
    elif '_cllora' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "cllora" :
            from backbone import vit_cllora
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                use_distillation = args["use_distillation"],
                use_block_weight = args["use_block_weight"],
                msa_adapt = args["msa_adapt"],
                msa = args["msa"],
                specfic_pos = args["specfic_pos"],
                general_pos = args["general_pos"],
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_cllora":
                model = vit_cllora.vit_base_patch16_224_cllora(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_cllora":
                model = vit_cllora.vit_base_patch16_224_in21k_cllora(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_memo' in name:
        if args["model_name"] == "memo":
            from backbone import vision_transformer_memo
            _basenet, _adaptive_net = timm.create_model("vit_base_patch16_224_memo", pretrained=True, num_classes=0)
            _basenet.out_dim = 768
            _adaptive_net.out_dim = 768
            return _basenet, _adaptive_net
    # SSF 
    elif '_ssf' in name:
        if args["model_name"] == "adam_ssf":
            from backbone import vision_transformer_ssf
            if name == "pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_ssf":
                model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"] == "adam_vpt":
            from backbone.vpt import build_promptmodel
            if name == "pretrained_vit_b16_224_vpt":
                basicmodelname = "vit_base_patch16_224" 
            elif name == "pretrained_vit_b16_224_in21k_vpt":
                basicmodelname = "vit_base_patch16_224_in21k"
            
            print("modelname,", name, "basicmodelname", basicmodelname)
            VPT_type = "Deep"
            if args["vpt_type"] == 'shallow':
                VPT_type = "Shallow"
            Prompt_Token_num = args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "sema":
            from backbone import vit_sema
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                exp_threshold=args["exp_threshold"],
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args["rd_dim"],
                buffer_size=args["buffer_size"],
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vit_sema.vit_base_patch16_224_sema(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vit_sema.vit_base_patch16_224_in21k_sema(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        if args["model_name"] == "adam_adapter" :
            from backbone import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # L2P
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            # print('args!!!!!!!',args)
            from backbone import vision_transformer_l2p
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # dualprompt
    elif '_dualprompt' in name:
        if args["model_name"] == "dualprompt":
            from backbone import vision_transformer_dual_prompt
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
                use_g_prompt=args["use_g_prompt"],
                g_prompt_length=args["g_prompt_length"],
                g_prompt_layer_idx=args["g_prompt_layer_idx"],
                use_prefix_tune_for_g_prompt=args["use_prefix_tune_for_g_prompt"],
                use_e_prompt=args["use_e_prompt"],
                e_prompt_layer_idx=args["e_prompt_layer_idx"],
                use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_e_prompt"],
                same_key_value=args["same_key_value"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # Coda_Prompt
    elif '_coda_prompt' in name:
        if args["model_name"] == "coda_prompt":
            from backbone import vision_transformer_coda_prompt
            model = timm.create_model(args["backbone_type"], pretrained=args["pretrained"],num_classes=0)
            # model = vision_transformer_coda_prompt.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
            #                 num_heads=12, ckpt_layer=0,
            #                 drop_path_rate=0)
            # from timm.models import vit_base_patch16_224
            # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # del load_dict['head.weight']; del load_dict['head.bias']
            # model.load_state_dict(load_dict)
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
        self.heads_w = []
        self.heads_b = []
        #self.heads = []
        self.head_warm_start = args.get('head_warm_start',False)
        self.integrate_old_heads = False
        self.head_weighting_coefs = []

    def old_head_integration(self, use_old_heads=True):
        self.integrate_old_heads = use_old_heads

    def set_head_weighting(self, weighting_coefs=None):
        self.head_weighting_coefs = weighting_coefs
    
    def save_fc(self, filename, task_id):

        # pass
        #self.heads.append(copy.deepcopy(self.fc))
        torch.save(self.fc.weight.detach(), filename + 'CLs_weight'+str(task_id)+'.pt')
        torch.save(self.fc.bias.detach(), filename + 'CLs_bias'+str(task_id)+'.pt')


    def load_fc(self, task_id):
        print('task_id', task_id)
        self.fc = self.generate_fc(self.feature_dim, (task_id+1)*10).cuda()
        for i in range(task_id+1):
            for j in range(i,task_id+1):
                temp_weights = torch.load('CLs_weight'+str(j)+'.pt')
                temp_bias = torch.load('CLs_bias'+str(j)+'.pt')
                # the following 10 means the incremental classes
                if j == i:
                    self.fc.weight.data[i*10: i*10+10] = temp_weights.data[i*10: i*10+10] 
                    self.fc.bias.data[i*10: i*10+10] = temp_bias.data[i*10:i*10+10].cuda()
                else:
                    self.fc.weight.data[i*10: i*10+10] += temp_weights.data[i*10: i*10+10].cuda()
                    self.fc.bias.data[i*10: i*10+10] += temp_bias.data[i*10: i*10+10].cuda()
            self.fc.weight.data = self.fc.weight.data/(task_id+1-i)
            self.fc.bias.data = self.fc.bias.data/(task_id+1-i)
        torch.save(self.fc.weight, 'CLs_weight'+str(task_id)+'.pt')
        torch.save(self.fc.bias, 'CLs_bias'+str(task_id)+'.pt')


    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features #10
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            #if tt_merge:
            #    self.heads_b.append(copy.deepcopy(self.fc.bias.data))
            #    self.heads_w.append(copy.deepcopy(self.fc.weight.data))
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            if self.head_warm_start:
                warmstart_weight = copy.deepcopy(self.fc.weight.data)
                print(warmstart_weight.shape)
                warmstart_bias = copy.deepcopy(self.fc.bias.data)
                last = nb_classes - nb_output
                print(last)
                fc.weight.data[nb_output:] = warmstart_weight[-last:]
                fc.bias.data[nb_output:] = warmstart_bias[-last:]

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x, ortho_loss=False, eval=False):
        if eval:
            out = self.backbone(x, eval=True)
            out.update({"features": x})
            return out
        else:
            if self.model_type == 'cnn':
                x = self.backbone(x)
                out = self.fc(x["features"])
                out.update(x)

            elif ortho_loss:
                x,ortho_loss = self.backbone(x, loss=True)
                out = self.fc(x)
                out.update({"features": x})
                return out, ortho_loss
            else:
                x = self.backbone(x)
                # if self.integrate_old_heads: #### for including old heads during training of new -> train merge
                #     out = self.fc(x)
                #     print(len(self.head_weighting_coefs))
                #     print(len(self.heads_w))
                #     with torch.no_grad():
                #         for i in range(len(self.heads_w)):
                #             out['logits'] += F.linear(x, self.heads_w[i]*self.head_weighting_coefs[i], self.heads_b[i]*self.head_weighting_coefs[i])
                # else:
                out = self.fc(x)
                out.update({"features": x})
                return out

            if hasattr(self, "gradcam") and self.gradcam:
                out["gradcam_gradients"] = self._gradcam_gradients
                out["gradcam_activations"] = self._gradcam_activations
                return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
            forward_hook
        )


class MingleNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
        self.heads_w = []
        self.heads_b = []
        #self.heads = []
        self.head_warm_start = args.get('head_warm_start',False)
        self.integrate_old_heads = False
        self.head_weighting_coefs = []

    
    def save_fc(self, filename, task_id):

        # pass
        #self.heads.append(copy.deepcopy(self.fc))
        torch.save(self.fc.weight.detach(), filename + 'CLs_weight'+str(task_id)+'.pt')
        torch.save(self.fc.bias.detach(), filename + 'CLs_bias'+str(task_id)+'.pt')


    def load_fc(self, task_id):
        print('task_id', task_id)
        self.fc = self.generate_fc(self.feature_dim, (task_id+1)*10).cuda()
        for i in range(task_id+1):
            for j in range(i,task_id+1):
                temp_weights = torch.load('CLs_weight'+str(j)+'.pt')
                temp_bias = torch.load('CLs_bias'+str(j)+'.pt')
                # the following 10 means the incremental classes
                if j == i:
                    self.fc.weight.data[i*10: i*10+10] = temp_weights.data[i*10: i*10+10] 
                    self.fc.bias.data[i*10: i*10+10] = temp_bias.data[i*10:i*10+10].cuda()
                else:
                    self.fc.weight.data[i*10: i*10+10] += temp_weights.data[i*10: i*10+10].cuda()
                    self.fc.bias.data[i*10: i*10+10] += temp_bias.data[i*10: i*10+10].cuda()
            self.fc.weight.data = self.fc.weight.data/(task_id+1-i)
            self.fc.bias.data = self.fc.bias.data/(task_id+1-i)
        torch.save(self.fc.weight, 'CLs_weight'+str(task_id)+'.pt')
        torch.save(self.fc.bias, 'CLs_bias'+str(task_id)+'.pt')


    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features #10
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            #if tt_merge:
            #    self.heads_b.append(copy.deepcopy(self.fc.bias.data))
            #    self.heads_w.append(copy.deepcopy(self.fc.weight.data))
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            if self.head_warm_start:
                warmstart_weight = copy.deepcopy(self.fc.weight.data)
                print(warmstart_weight.shape)
                warmstart_bias = copy.deepcopy(self.fc.bias.data)
                last = nb_classes - nb_output
                print(last)
                fc.weight.data[nb_output:] = warmstart_weight[-last:]
                fc.bias.data[nb_output:] = warmstart_bias[-last:]

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x, ortho_loss=False, eval=False):
        if eval:
            out = self.backbone(x, eval=True)
            out.update({"features": x})
            return out
        else:
            if self.model_type == 'cnn':
                x = self.backbone(x)
                out = self.fc(x["features"])
                out.update(x)
                return out

            elif ortho_loss:
                x,ortho_loss = self.backbone(x, loss=True)
                out = self.fc(x)
                out.update({"features": x})
                return out, ortho_loss
            else:
                x = self.backbone(x)
                # if self.integrate_old_heads: #### for including old heads during training of new -> train merge
                #     out = self.fc(x)
                #     print(len(self.head_weighting_coefs))
                #     print(len(self.heads_w))
                #     with torch.no_grad():
                #         for i in range(len(self.heads_w)):
                #             out['logits'] += F.linear(x, self.heads_w[i]*self.head_weighting_coefs[i], self.heads_b[i]*self.head_weighting_coefs[i])
                # else:
                out = self.fc(x)
                out.update({"features": x})
                return out

            if hasattr(self, "gradcam") and self.gradcam:
                out["gradcam_gradients"] = self._gradcam_gradients
                out["gradcam_activations"] = self._gradcam_activations
                return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num  ==  1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc

class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.backbones) == 0:
            self.backbones.append(get_backbone(self.args, self.pretrained))
        else:
            self.backbones.append(get_backbone(self.args, self.pretrained))
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out

class SEMAVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.fc = None
        self.args = args

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        out = self.backbone(x)
        x = out["features"]
        out.update({"logits": self.fc(x)})
        return out
    
    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features #10
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            #if tt_merge:
            #    self.heads_b.append(copy.deepcopy(self.fc.bias.data))
            #    self.heads_w.append(copy.deepcopy(self.fc.weight.data))
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            if self.args.get("head_warm_start",False):
                warmstart_weight = copy.deepcopy(self.fc.weight.data)
                print(warmstart_weight.shape)
                warmstart_bias = copy.deepcopy(self.fc.bias.data)
                last = nb_classes - nb_output
                print(last)
                fc.weight.data[nb_output:] = warmstart_weight[-last:]
                fc.bias.data[nb_output:] = warmstart_bias[-last:]

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5))
        nn.init.zeros_(fc.bias)
        #fc = SimpleLinear(in_dim, out_dim)
        return fc

# l2p and dualprompt
class PromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(PromptVitNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        if args["get_original_backbone"]:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None
            
    def get_original_backbone(self, args):
        return timm.create_model(
            args["backbone_type"],
            pretrained=args["pretrained"],
            num_classes=args["nb_classes"],
            drop_rate=args["drop"],
            drop_path_rate=args["drop_path"],
            drop_block_rate=None,
        ).eval()

    def update_fc(self, task_id, nb_classes):
        #fc = self.generate_fc(self.feature_dim, nb_classes)
        
        #nb_output = self.fc.out_features #10
        logging.info('Head warm starting...')
        warmstart_weight = copy.deepcopy(self.backbone.head.weight.data)[(task_id-1)*nb_classes:task_id*nb_classes]
        warmstart_bias = copy.deepcopy(self.backbone.head.bias.data)[(task_id-1)*nb_classes:task_id*nb_classes]
        #if tt_merge:
        #    self.heads_b.append(copy.deepcopy(self.fc.bias.data))
        #    self.heads_w.append(copy.deepcopy(self.fc.weight.data))
        self.backbone.head.weight.data[task_id*nb_classes:(task_id+1)*nb_classes] = warmstart_weight
        self.backbone.head.bias.data[task_id*nb_classes:(task_id+1)*nb_classes] = warmstart_bias
        
    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x

# coda_prompt
class CodaPromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        self.fc = nn.Linear(768, args["nb_classes"])
        self.prompt = CodaPrompt(768, args["nb_tasks"], args["prompt_param"])

    # pen: get penultimate features  
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:,0,:]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:,0,:]
        else:
            out, _ = self.backbone(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

    def update_fc(self, task_id, nb_classes):
        #fc = self.generate_fc(self.feature_dim, nb_classes)
        
        #nb_output = self.fc.out_features #10
        warmstart_weight = copy.deepcopy(self.fc.weight.data)[(task_id-1)*nb_classes:task_id*nb_classes]
        warmstart_bias = copy.deepcopy(self.fc.bias.data)[(task_id-1)*nb_classes:task_id*nb_classes]
        #if tt_merge:
        #    self.heads_b.append(copy.deepcopy(self.fc.bias.data))
        #    self.heads_w.append(copy.deepcopy(self.fc.weight.data))
        self.fc.weight.data[task_id*nb_classes:(task_id+1)*nb_classes] = warmstart_weight
        self.fc.bias.data[task_id*nb_classes:(task_id+1)*nb_classes] = warmstart_bias



class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        
        # no need the backbone.
        
        print('Clear the backbone in MultiBranchCosineIncrementalNet, since we are using self.backbones with dual branches')
        self.backbone=torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args=args
        
        if 'resnet' in args['backbone_type']:
            self.model_type='cnn'
        else:
            self.model_type='vit'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]       

        features = torch.cat(features, 1)
        # import pdb; pdb.set_trace()
        out = self.fc(features)
        out.update({"features": features})
        return out

    
    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_ssf','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs)) #pretrained model without scale
        elif 'vpt' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_vpt','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs)) #pretrained model without vpt
        elif 'adapter' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_adapter','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs)) #pretrained model without adapter
        else:
            self.backbones.append(get_backbone(self.args)) #the pretrained model itself

        self.backbones.append(tuned_model.backbone) #adappted tuned model
    
        self._feature_dim = self.backbones[0].out_dim * len(self.backbones) 
        self.fc=self.generate_fc(self._feature_dim,self.args['init_cls'])


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.backbones.append(get_backbone(self.args, self.pretrained))
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.TaskAgnosticExtractor , _ = get_backbone(args, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
        
    def update_fc(self,nb_classes):
        _ , _new_extractor = get_backbone(self.args, self.pretrained)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            # logging.info(self.AdaptiveExtractors[-1])
            self.out_dim=self.AdaptiveExtractors[-1].out_dim        
        fc = self.generate_fc(self.feature_dim, nb_classes)             
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc
    
class OurNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.nb_classes_task = args['nb_classes']
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = False
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.fc_list = nn.ModuleList()
        self.fc_list_task = nn.ModuleList()
        self.adapter_list = nn.ModuleList()
        self.init_proto = None

            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    @property
    def feature_dim(self):

        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)

    def update_fc_task(self):
        self.proxy_fc_task = self.generate_fc(self.out_dim, 1).to(self._device)
        self.fc_list_task.append(self.proxy_fc_task.requires_grad_(True))

    def update_fc(self, nb_classes):
        self._cur_task += 1
        
        if self._cur_task == 0:
            print('gernerate prot_fc ', self._cur_task)
            self.proxy_fc = self.generate_fc(self.out_dim, self.nb_classes_task).to(self._device)
        else:
            print('gernerate prot_fc ', self._cur_task)
            self.proxy_fc = self.generate_fc(self.out_dim, self.nb_classes_task).to(self._device)
        init_proto = self.generate_fc(self.out_dim, nb_classes).to(self._device)

        if self.init_proto is not None:
            old_nb_classes = self.init_proto.out_features
            weight = copy.deepcopy(self.init_proto.weight.data)
            init_proto.weight.data[: old_nb_classes, :] = nn.Parameter(weight)
        del self.init_proto
        self.init_proto = init_proto

        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[: old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        self.fc.requires_grad_(False)

    def add_fc(self):
        self.fc_list.append(self.proxy_fc.requires_grad_(False))
        del self.proxy_fc

    def remove_fc_task(self):
        self.fc_list_task.requires_grad_(False)
        del self.proxy_fc_task

    def remove_fc_init(self):
        self.init_proto_list.append(self.init_proto)
        del self.init_proto
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinearFeature(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward_kd(self, x, t_idx):
        x_new, x_teacher = self.backbone.forward_general_cls(x, t_idx)
        out_new, out_teacher = self.proxy_fc(x_new), self.proxy_fc(x_teacher)
        return  out_new, out_teacher

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
            out.update({"features": x})
            return out
        else:
            x_input = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_diagonal(x_input, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.nb_classes_task, inc=self.nb_classes_task, use_init_ptm=self.use_init_ptm, beta=self.beta)

            out.update({"features": x_input})
            return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())