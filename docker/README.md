# Docker

## Image Size 
Current resulting image size is 12GB. 
To decrease image size drastically we need to step away from PyTorch entirely. 

## Build
If building on AWS machine make sure you have done the steps listed in the main README [here](../README.md)

docker run --runtime=nvidia -rm 

## GUI Support 
GUI support is switched off by default, though dependencies needed are baked into the building docker stages. 
For **COLMAP** switch `-DGUI_ENABLED=OFF` to `ON`  
For **OpenMVS** switch `DOpenMVS_USE_QT=OFF` to `ON`
*Currently the GUI support is untested. It works flawlessy in the `.devcontainer` environment. For GUI support I suppose switching to devcontainer environment - maybe having one devconatainer environment available on office Computers for debugging customer output or advancing pipeline functionality*
