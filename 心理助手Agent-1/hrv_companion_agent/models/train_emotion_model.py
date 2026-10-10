"""Current independent-label training entry. Auto-labelled memory training retired."""
if __package__:
    from .train_multimodal import main
else:
    from train_multimodal import main
if __name__ == '__main__':
    main()
