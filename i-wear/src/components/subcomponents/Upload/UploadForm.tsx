"use client";

import React, { SyntheticEvent } from "react";

interface UploadFormProps {
  image: string | null;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const UploadForm: React.FC<UploadFormProps> = ({ image, onFileChange }) => {
const POST_URL = "http://localhost:8080/predict";

  const formSubmitHandler = async (e: React.SyntheticEvent) => {
    e.preventDefault();
    console.log(image)
    const data = await fetch(POST_URL, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({image}),
    });
    const response = await data.json();
    console.log(response);
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
