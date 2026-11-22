import os
import shutil


def extract_images(folder_path):

    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith(
            ('.png', '.jpg', '.jpeg')
        )
    ]

    print("Images found:")
    print(image_files)

    if not image_files:
        print("No images found.")
        return

    dead_prefix = ['7', '16', '23', '49', '58', '89', '91', '96', '105', '108', '109', '111', '112', '124']

    male_prefix = ['2', '9', '10', '15', '19', '24', '31', '36', '39', '42', '44', '45', '52', '53', '59', '60', '63', '65', '66', '67', '68', '71', '73', '76', '83', '85', '86',
        '87', '94', '95', '98', '102', '103', '107', '113', '114', '116', '119', '120', '134']

    female_prefix = [
       '1', '3', '4', '5', '6', '8', '11', '13', '17', '18', '25', '26', '27', '29', '30', '32', '38', '40', '41', '46', '48', '51', '56', '57', '61', '62', '64', '69', '70', '72', '74', '75', '78', '79',
        '80', '81', '82', '88', '92', '97', '99', '100', '115', '121', '123', '125', '126', '127', '128', '129', '131']


    male_folder = os.path.join(
        folder_path,
        "Male"
    )

    female_folder = os.path.join(
        folder_path,
        "Female"
    )

    dead_folder = os.path.join(
        folder_path,
        "Dead"
    )

    os.makedirs(male_folder, exist_ok=True)
    os.makedirs(female_folder, exist_ok=True)
    os.makedirs(dead_folder, exist_ok=True)


    male_count = 0
    female_count = 0
    dead_count = 0


    for file_name in image_files:


        prefix = ''.join(
            ch for ch in file_name
            if ch.isdigit()
        )

        if not prefix:
            print(f"Skipping a filename without digits: {file_name}")
            continue


        print(f"{file_name} -> egg ID {prefix}")


        if prefix in dead_prefix:

            src = os.path.join(
                folder_path,
                file_name
            )

            dst = os.path.join(
                dead_folder,
                file_name
            )

            if not os.path.exists(dst):

                shutil.copyfile(src, dst)

                dead_count += 1

                print(f"Copied to Dead: {file_name}")


        elif prefix in male_prefix:

            src = os.path.join(
                folder_path,
                file_name
            )

            dst = os.path.join(
                male_folder,
                file_name
            )

            if not os.path.exists(dst):

                shutil.copyfile(src, dst)

                male_count += 1

                print(f"Copied to Male: {file_name}")


        elif prefix in female_prefix:

            src = os.path.join(
                folder_path,
                file_name
            )

            dst = os.path.join(
                female_folder,
                file_name
            )

            if not os.path.exists(dst):

                shutil.copyfile(src, dst)

                female_count += 1

                print(f"Copied to Female: {file_name}")


        else:

            print(f"No label for {file_name}. Skipping.")


    print("\nSorting complete.")
    print(f"Male: {male_count} images")
    print(f"Female: {female_count} images")
    print(f"Dead: {dead_count} images")


folder_path = r"C:\Users\DELL\Desktop\B3-D3"

extract_images(folder_path)
