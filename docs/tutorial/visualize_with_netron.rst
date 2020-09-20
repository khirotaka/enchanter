Visualize with Netron
=========================

::

    from enchanter.addons.layers import AutoEncoder
    x = torch.randn(1, 32)  # [N, in_features]
    model = AutoEncoder([32, 16, 8, 2])
    with_netron(model, (x, ))


..  image:: assets/netron_viewer.png
    :scale: 30 %
    :alt: Netron Viewer
    :align: center
