"use client";

import React from "react";

interface UploadFormProps {
  image: string | null;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setFaceStructure: React.Dispatch<React.SetStateAction<string | null>>
  setIsRequesting: React.Dispatch<React.SetStateAction<boolean>>
}

const UploadForm: React.FC<UploadFormProps> = ({ image, onFileChange, setIsLoading, setFaceStructure, setIsRequesting }) => {
const POST_URL = "http://localhost:8080/predict";

  const formSubmitHandler = async (e: React.SyntheticEvent) => {
    e.preventDefault();
    setFaceStructure(null);
    setIsRequesting(true);
    setIsLoading(true);
    const data = await fetch(POST_URL, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({image}),
    });
    const response = await data.json();
    setIsLoading(false);
    if (response.response_code != 200) {
        return
    }
    setFaceStructure(response.prediction);
    setTimeout(() => {
        setIsRequesting(false);
        setFaceStructure(null);

    }, 10000);
};


  return (
    <div>
      <h2>Upload Photo</h2>
      <input type="file" accept="image/*" onChange={onFileChange} />
      {image && (
        <div>
          <img src={image} alt="Uploaded" />
        </div>
      )}
      <button type="submit" onClick={formSubmitHandler}>Request</button>
    </div>
  );
};

export default UploadForm;
