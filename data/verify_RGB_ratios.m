% This program computes the normalized spectral ratios for each (linear) image
% in a given folder. This is used to verify that no other non-linear 
% transformations have been applied to sRGB images.



% user input: folder path
folder_path = input('Enter the folder path: ', 's');

% loop over the png files
png_files = dir(fullfile(folder_path, '*.png'));
for file_idx = 1:length(png_files)

    fn = fullfile(folder_path, png_files(file_idx).name);
    disp(png_files(file_idx).name)    

    try
        
        % read the sRGB image and transform to linear RGB
        img_srgb = imread(fn);
        img_rgb = double(img_srgb) / 255.0;
        img_rgb = ...
            (img_rgb <= 0.04045) .* (img_rgb / 12.92) + ...
            (img_rgb > 0.04045)  .* ((img_rgb + 0.055) / 1.055) .^ 2.4;

        % store computed normalized spectral ratios
        normalized_spectral_ratios = [];

        % display the linear RGB image
        figure;
        imshow(img_rgb);
        title('Select two points: bright and dark');

        % until the user closes the window...
        while ishandle(gcf)

            try

                % user selects two points of the same color: 
                % bright (non-shadow), then dark (in shadow)
                [x, y] = ginput(2);
                x = round(x);
                y = round(y);
    
                % extract selected points
                bright = reshape(img_rgb(y(1), x(1), :), [1, 3]);
                dark   = reshape(img_rgb(y(2), x(2), :), [1, 3]);
    
                % store the normalized spectral ratio
                spectral_ratio = dark ./ (bright - dark);
                normalized_spectral_ratio = spectral_ratio / norm(spectral_ratio);
                normalized_spectral_ratios = cat(1, normalized_spectral_ratios, normalized_spectral_ratio);
            
            catch
                % the user closed the window... or something went wrong
                break;
            end  % end try

        end  % done selecting bright/dark pairs

        % display the normalized spectral ratios, and the variance between
        % them
        disp(['Normalized Spectral Ratios for ', png_files(file_idx).name, ':']);
        disp(normalized_spectral_ratios);
        variance = var(normalized_spectral_ratios);
        disp(['Variance between normalized spectral ratios: ', num2str(variance)]);
        disp(' ')
    
    catch
        disp(['Error reading the image ', png_files(file_idx).name, '.']);
    end  % end try

    close all;

end  % loop to next image