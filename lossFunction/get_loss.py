from bce_loss_with_one_linear_discriminator import loss as bce_loss
from cca_ssg_loss import loss as cca_ssg_loss
from loacl_global_loss import loss as loacl_global_loss
from grace_loss import loss as grace_loss


def get_loss(objective, task_type):
    if callable(objective):
        return objective

    assert objective in ['cca_ssg', 'gca', 'grace', 'graphcl', 'dgi', 'mvgrl.py']
    assert task_type in ['node', 'graph']

    if task_type == 'node':
        return {
            'cca_ssg': cca_ssg_loss,
            'gca': grace_loss,
            'grace': grace_loss,
            'graphcl': bce_loss,
            'dgi': bce_loss,
            'mvgrl.py': bce_loss,
        }[objective]
    elif task_type == 'graph':
        return {
            'graphcl':loacl_global_loss,
            'mvgrl.py': loacl_global_loss
        }[objective]