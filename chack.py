def check_device():
    """Kullanılabilir cihazları kontrol eder ve en uygun cihazı seçer"""
    import torch

    print("\n=== CİHAZ BİLGİLERİ ===")
    print(f"CUDA kullanılabilir mi: {torch.cuda.is_available()}")
    print(f"MPS kullanılabilir mi: {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon GPU (MPS) kullanılıyor")
    else:
        device = 'cpu'
        print("CPU kullanılıyor")

    return device