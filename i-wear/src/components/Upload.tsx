"use client";

import React, { useState } from "react";
import UploadForm from "./subcomponents/Upload/UploadForm";
import Response from "./subcomponents/Upload/Response";

interface UploadProps {
  base64Image: string | null;
  setBase64Image: React.Dispatch<React.SetStateAction<string | null>>;
  isRequesting: boolean;
  setIsRequesting: React.Dispatch<React.SetStateAction<boolean>>;
  isLoading: boolean;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  faceStructure: string | null;
  setFaceStructure: React.Dispatch<React.SetStateAction<string | null>>;
  setProbabilities: React.Dispatch<React.SetStateAction<number[] | null>>;
}

const Upload: React.FC<UploadProps> = ({base64Image, setBase64Image, isRequesting,
  setIsRequesting,
  isLoading,
  setIsLoading,
  faceStructure,
  setFaceStructure, setProbabilities}) => {


  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result;
        if (typeof result === "string") {
          setBase64Image(result);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div>
      <UploadForm onFileChange={handleImageChange} image={base64Image} setFaceStructure={setFaceStructure} setIsLoading={setIsLoading} setIsRequesting={setIsRequesting} setProbabilities={setProbabilities} />
      <Response isLoading={isLoading} isRequesting={isRequesting} faceStructure={faceStructure} />
    </div>
  );
};

export default Upload;
