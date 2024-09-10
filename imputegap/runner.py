import imputegap


def display_title(title="Master Thesis", aut="Quentin Nater", lib="ImputeLIB",
                  university="University Fribourg - exascale infolab", file="runner"):
    print("=" * 100)
    print(f"{title} : {aut}")
    print("=" * 100)
    print(f"    {lib} - {university}")
    print("=" * 100)
    print(f"    {file}.py")
    print("=" * 100)


if __name__ == '__main__':
    display_title()

    print("\n", "_"*95, "end")


