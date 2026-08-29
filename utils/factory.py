def get_model(model_name, args):
    name = model_name.lower()
    if name == "ttime":
        from models.ttime import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "ttime-peft":
        from models.ttime_peft import Learner
    else:
        assert 0
    
    return Learner(args)
