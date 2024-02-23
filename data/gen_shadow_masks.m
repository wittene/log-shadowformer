
% SETUP
ROOT = "C:\Users\eawln\Documents\Edu\NU\25\MS Project\srijha-sr-dataset\compressed\";
CLEAN_DIR = fullfile(ROOT, "clean/");
SHADOW_DIR = fullfile(ROOT, "shadow/");
MASK_DIR = fullfile(ROOT, "mask/");

DIFF_THRESH = 50;
SE = strel('disk', 40);

% Loop through images in shadow directory
shadow_files = dir(fullfile(SHADOW_DIR, "*.JPG"));
for i = 1:numel(shadow_files)
    % Get file names
    shadow_fn = fullfile(SHADOW_DIR, shadow_files(i).name);
    mask_fn = fullfile(MASK_DIR, shadow_files(i).name);
    if exist(mask_fn, "file") == 2
        % if mask already exists, skip
        continue
    end
    [~, img_name, img_ext] = fileparts(shadow_fn);
    img_name_parts = strsplit(img_name, {'-', '.'});
    scene_id = img_name_parts(1);
    clean_fn = fullfile(CLEAN_DIR, strcat(scene_id, '-1', img_ext));
    % Load shadow and clean images
    im_shadow = imread(shadow_fn);
    im_clean = imread(clean_fn);    
    % Manual mask adjustment loop
    done = false;
    while ~done
        f = figure;
        imshow(im_shadow);
        title(img_name);
        roi = drawassisted;
        im_mask = createMask(roi);
        imshow(im_mask);
        waitfor(f);
        confirmation = input('Confirm? (Y/N)  > ', 's');
        if strcmpi(confirmation, 'Y')
            done = true;
        end
    end    
    % Save
    fprintf('(%d) Saving to %s\n', i, mask_fn);
    imwrite(im_mask, mask_fn);
end

% % Estimate the shadow region using diff
% im_diff = imabsdiff(im_shadow, im_clean);
% % Binarize using threshold
% im_binary = true(size(im_diff, 1), size(im_diff, 2));
% for c = 1:3
%     im_binary = im_binary & (im_diff(:, :, c) > DIFF_THRESH);
% end
% % Morphological filtering
% im_filtered = bwareaopen(im_binary, 50);
% im_filtered = imclose(im_filtered, SE);
