"use client";

import React, { useState } from "react";
import UploadForm from "./subcomponents/Upload/UploadForm";

const Upload: React.FC = () => {
  const [base64Image, setBase64Image] = useState<string | null>(null);

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
      <UploadForm onFileChange={handleImageChange} image={base64Image} />
    </div>
  );
};

export default Upload;
