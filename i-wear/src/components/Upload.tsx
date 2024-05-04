"use client";

import React, { useState } from "react";
import UploadForm from "./subcomponents/Upload/UploadForm";
import Response from "./subcomponents/Upload/Response";

const Upload: React.FC = () => {
  const [base64Image, setBase64Image] = useState<string | null>(null);
  const [isRequesting, setIsRequesting] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [faceStructure, setFaceStructure] = useState<string | null>(null);

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
      <UploadForm onFileChange={handleImageChange} image={base64Image} setFaceStructure={setFaceStructure} setIsLoading={setIsLoading} setIsRequesting={setIsRequesting} />
      <Response isLoading={isLoading} isRequesting={isRequesting} faceStructure={faceStructure} />
    </div>
  );
};

export default Upload;
