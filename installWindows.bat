mkdir tmpInstallHtMouse
copy environment.yml tmpInstallHtMouse
cd tmpInstallHtMouse

powershell
curl -fSL https://repo.anaconda.com/miniconda/Miniconda2-latest-Windows-x86_64.exe > miniconda_installer.exe

miniconda_installer.exe

conda env create --file=environment.yml

echo "Removing installation files."
cd ..
rmdir /s /q tmpInstallHtMouse
echo "Install completed successfully."