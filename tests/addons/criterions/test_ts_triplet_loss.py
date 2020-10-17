import torch
import enchanter.addons.criterions as C
import enchanter


def test_generate_sample_indices_1():
    n_rand_samples: int = 10
    batch_size: int = 32
    length: int = 100

    begin_batches, len_anchor, end_pos, len_pos_neg, begin_neg_samples = C.generate_sample_indices(
        n_rand_samples, batch_size, length
    )

    assert (length > begin_batches).all()
    assert length > len_pos_neg
    assert length > len_anchor
    assert (length > end_pos).all()
    assert (length > begin_neg_samples.numpy()).all()


def test_generate_sample_indices_2():
    is_pass = False

    n_rand_samples: int = 10
    batch_size: int = 32
    length: int = -1
    try:
        begin_batches, len_anchor, end_pos, len_pos_neg, begin_neg_samples = C.generate_sample_indices(
            n_rand_samples, batch_size, length
        )
    except ValueError:
        is_pass = True

    assert is_pass


def test_generate_sample_indices_3():
    is_pass = False

    n_rand_samples: int = -1
    batch_size: int = 32
    length: int = 1
    try:
        begin_batches, len_anchor, end_pos, len_pos_neg, begin_neg_samples = C.generate_sample_indices(
            n_rand_samples, batch_size, length
        )
    except ValueError:
        is_pass = True

    assert is_pass


def test_generate_sample_indices_4():
    is_pass = False

    n_rand_samples: int = 10
    batch_size: int = -1
    length: int = 10
    try:
        begin_batches, len_anchor, end_pos, len_pos_neg, begin_neg_samples = C.generate_sample_indices(
            n_rand_samples, batch_size, length
        )
    except ValueError:
        is_pass = True

    assert is_pass


def test_generate_sample_indices_5():
    is_pass = False

    n_rand_samples: int = -1
    batch_size: int = -1
    length: int = -1
    try:
        begin_batches, len_anchor, end_pos, len_pos_neg, begin_neg_samples = C.generate_sample_indices(
            n_rand_samples, batch_size, length
        )
    except ValueError:
        is_pass = True

    assert is_pass
