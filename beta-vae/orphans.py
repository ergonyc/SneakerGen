# orphan functions


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    scale = tr.random.uniform([], minval=1, maxval=1.5, dtype=tf.float32, seed=None, name=None)
    tf.image.resize_images(tf.expand_dims(img, 0), tf.cast([H * scale, W * scale], tf.int32))

    tf.keras.preprocessing.image.random_zoom(
        image,
        zoom_range=[1.0, 1.1],
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode="nearest",
        cval=0.0,
        interpolation_order=1,
    )
    return image, label


def augment_and_batch(dataset, batch_size):
    # use zoom to make the image square and white border.
    tf.keras.preprocessing.image.random_zoom(
        x,
        zoom_range,
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode="nearest",
        cval=0.0,
        interpolation_order=1,
    )
    dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset


def test_batch(dataset, batch_size):
    dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset


def load_img_and_preprocess(filename, vox_size=64):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    shape_f = tf.cast(tf.shape(image), tf.float32)
    print(f"shape = {shape_f}, len = {len(shape_f)}")
    if len(shape_f) == 3:
        initial_height, initial_width = (
            shape_f[0],
            shape_f[1],
        )  # at this point we don't have teh extra dimension...
    else:
        print("ERRRORRRRRR")
    print(f"height = {initial_height},w = {initial_width}")

    image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
    # image = make_square(image)

    # # we want to center in a box which is square and 5% bigger than the initial width...
    # # x1 = 0 - 0.025 * (initial_width - 1)
    # # x2 = 1 + 0.025 * (initial_width - 1)
    # # y1 = 0 - 0.025 * (initial_width - 1) - (initial_width - initial_height) / 2.0
    # # y2 = 1 + 0.025 * (initial_width - 1) + (initial_width - initial_height) / 2.0

    x1 = 0.0  # - 0.025 * (initial_width - 1)
    x2 = 1.0  # + 0.025 * (initial_width - 1)
    y1 = 0.0 - 0.5 * (initial_width - initial_height) / initial_width
    y2 = 1.0 + 0.5 * (initial_width - initial_height) / initial_width
    box = [y1, x1, y2, x2]
    image = tf.image.crop_and_resize(
        tf.expand_dims(image, 0),
        boxes=[box],
        box_indices=[0],
        crop_size=[vox_size, vox_size],
        extrapolation_value=1.0,
    )
    # # image = tf.image.crop_and_resize(
    #     image, offset_height, offset_width, target_size, target_size, constant_values=pad_value
    # )
    label = tf.constant(-1, tf.int32)

    return image, label

    # shape = im.get_shape()
    # target_size = max(shape)
    # image = tf.image.pad_to_bounding_box(
    #     image, offset_height, offset_width, target_size, target_size, constant_values=pad_value
    # )

    # im = tf.image.crop_and_resize(
    #     im, boxes, box_indices, crop_size, method="bilinear", extrapolation_value=0, name=None
    # )
    # im = tf.image.resize_with_crop_or_pad(im, target_size, target_size)
    # # im = tf.image.resize_images(im, [vox_size, vox_size])
    # im = pad_square(im, vox_size, pad_value=255.0)

    # boxes = tf.Variable([[100, 100, 300, 300]])
    # box_ind = tf.Variable([0])
    # crop_size = tf.Variable([target_size, target_size])
    # processed_img = tf.image.crop_and_resize(
    #     im,
    #     boxes=[[0.4529, 0.72, 0.4664, 0.7358]],
    #     crop_size=[target_size, target_size],
    #     box_ind=[0],
    #     extrapolation_value=255.0,
    # )
    # # b = tf.image.crop_and_resize(img,[[0.5,0.1,0.9,0.5]],[0],[50,50])
    # c = tf.image.crop_and_resize(img_, boxes, box_ind, crop_size)

    # image = tf.random.normal(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    # boxes = tf.constant([1, 4], tf.int16)
    # box_indices
    # boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
    # box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    # output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
    # output.shape  # => (5, 24, 24, 3)

    # im = tf.image.resize_with_pad(im, vox_size, vox_size)
    # im = tf.image.resize_images(im, [vox_size, vox_size])
    # label = get_label(filename)
    # print(label)

    # return tf.cast(im, tf.float32) / 255.0, tf.constant(-1, tf.int32)


# all_voxs, all_mids = ut.loadData(cf_vox_size, cf_max_loads_per_cat, lg.vox_in_dir, cf_cat_prefixes)
# example at bottom to wrap a keras imagedatagenerator
def loadData(target_vox_size, vox_in_dir):

    files = glob.glob(os.path.join(vox_in_dir, "*/img/*"))
    ds = tf.data.Dataset.from_tensor_slices(files)

    foo = lambda x: load_img_and_preprocess(x, target_vox_size)
    ds = tf.data.Dataset.from_tensor_slices(files).map(foo, num_parallel_calls=AUTOTUNE)
    # this doesn't really work...
    # img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rescale=1.0, rotation_range=20, fill_mode="constant", cval=1.0
    # )
    # gen = img_gen.flow(ds)
    # dataset = tf.data.Dataset.from_generator(lambda: gen, output_types=(tf.float32, tf.int32))

    return ds


def loadAndPrepData(target_vox_size, vox_in_dir, cf_batch_size):
    ds = loadData(target_vox_size, vox_in_dir)

    TEST_FRAC = 20.0 / 100.0
    # train_dataset, ds = splitShuffleData(ds, TEST_FRAC)
    train_dataset, ds = splitShuffleData(ds, TEST_FRAC)
    val_dataset, test_dataset = splitData(ds, 0.5)

    train = (
        train_dataset.map(augment, num_parallel_calls=AUTOTUNE)  # The augmentation is added here.
        .batch(cf_batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    test = test_dataset.batch(cf_batch_size, drop_remainder=False)
    validate = val_dataset.batch(cf_batch_size, drop_remainder=False)

    return train, test, validate


def make_square(image, resize=False, pad_value=1.0):
    # returns a square image the size of the inital width
    # assume width > height
    shape_f = tf.cast(tf.shape(image), tf.float32)
    initial_height, initial_width = shape_f[1], shape_f[2]
    if resize:
        target_size = resize
    else:
        target_size = initial_width

    x1 = 0.0  # - 0.025 * (initial_width - 1)
    x2 = 1.0  # + 0.025 * (initial_width - 1)
    y1 = 0.0 - 0.5 * (initial_width - initial_height) / initial_width
    y2 = 1.0 + 0.5 * (initial_width - initial_height) / initial_width
    image = tf.image.crop_and_resize(
        image,
        boxes=[[y1, x1, y2, x2]],
        box_indices=[0],
        crop_size=[target_size, target_size],
        extrapolation_value=pad_value,
    )
    return image
