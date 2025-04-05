try:
    from rtamp import sum_as_string
    print(f"1 + 2 = {sum_as_string(1, 2)}")
    print("rtamp module imported successfully!")
except Exception as e:
    print(f"Import failed: {e}")