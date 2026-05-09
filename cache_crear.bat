@echo off
rem cache_crear.bat - Clear build caches but preserve any 'support' folders
setlocal

nset ROOT=%~dp0
echo Clearing build caches in "%ROOT%" (support folders will be preserved)

nrem Remove common build output dirs
if exist "%ROOT%build" (
  echo Removing: %ROOT%build
  rmdir /s /q "%ROOT%build"
)
if exist "%ROOT%dist" (
  echo Removing: %ROOT%dist
  rmdir /s /q "%ROOT%dist"
)
if exist "%ROOT%.pytest_cache" (
  echo Removing: %ROOT%.pytest_cache
  rmdir /s /q "%ROOT%.pytest_cache"
)

nrem Remove all __pycache__ dirs except those under a 'support' path
for /d /r "%ROOT%" %%D in (__pycache__) do (
  echo %%D | findstr /i "\\support\\" >nul
  if errorlevel 1 (
    echo Removing %%D
    rmdir /s /q "%%D"
  ) else (
    echo Skipping support cache: %%D
  )
)

nrem Delete compiled python files (.pyc, .pyo) except under any 'support' folders
for /r "%ROOT%" %%F in (*.pyc) do (
  echo %%F | findstr /i "\\support\\" >nul
  if errorlevel 1 del /f /q "%%F" 2>nul
)
for /r "%ROOT%" %%F in (*.pyo) do (
  echo %%F | findstr /i "\\support\\" >nul
  if errorlevel 1 del /f /q "%%F" 2>nul
)

necho Done. Cache cleared (support folders preserved).
endlocal
